import matplotlib.colors
from Levenshtein import distance as levenshtein_distance
import re
import pandas as pd
import numpy as np
def generate_xyz_values():
    results = []
    step = 0.1
    for x in range(11):  # Using a range from 0 to 100 to get values in the range [0, 1] with step 0.01
        for y in range(11):
            z = 10 - x - y
            if z >= 0:
                x_value = x *step
                y_value = y *step
                z_value = z *step
                if x_value + y_value + z_value == 1.0:
                    results.append((x_value, y_value, z_value))
    return results

def calculate_remine_consistency(old_rule, new_rule):
    length1 = len(old_rule)
    length2 = len(new_rule)
    distance = 1-levenshtein_distance(old_rule, new_rule) #adaption according to ER review
    if(distance<0):
        distance =0
    print(distance, length1, length2)
    return distance
def find_relational_rules(decision_rule, variable_names_used, variable_input):
    all_variables = []
    found = False
    count = 0
    for v in variable_input:
            if v in decision_rule:
                count += 1
            if count > 1:
                found = True
    if found:
        for v in variable_input:
                if v in decision_rule:
                    all_variables.append(v)
    return all_variables, bool(found)


def calculate_parsimony_new(decision_rule, variable_input, variable_names_used, x,y,z):
    def num_there(s):
        return any(i.isdigit() for i in s)
    decision_rule = decision_rule.replace('IF', '')
    decision_rule = re.split('AND | OR',decision_rule)
    d = dict.fromkeys(variable_names_used, 0)
    counter_relational = 0
    for rule in decision_rule:
        all_variables, found = find_relational_rules(rule, variable_names_used, variable_input)
        if found:
            counter_relational += 1
    for v in variable_names_used:
        if num_there(v):
            result = ''.join([i for i in v if not i.isdigit()])
            if result in d:
                d[result] = d[result] + 1
            else:
                d[result] = 1
    duplicates = 1
    duplicate_value = []
    for key, value in d.items():
        if int(value) > 1:
            duplicates += int(value)
            duplicate_value.append(key)
    engineered_score = count_engineered_features(variable_input, variable_names_used)/len(variable_names_used)
    parsimony = (1-engineered_score)*x + (1-counter_relational/len(variable_names_used))*y + (1/len(variable_names_used))*z

    return parsimony


def count_engineered_features(variables_input, variable_names_used):
    variable_matches = list(set(variable_names_used))
    variable_found_count = len(set(variable_matches) & set(variables_input))
    return len(variable_names_used)-variable_found_count

def calculate_interpretability_new(decision_rule, variable_input, variable_names_used, x, y, z):
    import regex as re

    word_score = 0
    sonderzeichen_score = 0
    decision_rule = decision_rule.replace('IF', '')
    decision_rule = re.split('AND | OR',decision_rule)
    all_variables = []
    for rule in decision_rule:
        all_variables, found = find_relational_rules(rule, variable_names_used, variable_input)
    variable_names_used.extend(all_variables)
    engineered_score = count_engineered_features(variable_input, variable_names_used)/len(variable_names_used) #count engineered variable

    all_variables_without_sonderzeichen = []
    new = []
    count = 0
    count_words = 0
    for var in variable_names_used:
        search= re.findall(pattern="[^a-zA-Z]", string=var)
        count += len(var)
        if search:
            sonderzeichen_score += len(search)
            count_words += 1
            new.extend(re.split(pattern="[^a-zA-Z]+", string=var))
        elif re.findall(pattern="(\w+)([A-Z]\w+)", string=var):
            split = re.findall(pattern="(\w+)([A-Z]\w+)", repl=",", string=var)
            new.append(split[0][0])
            new.append(split[0][1])
        else:
            all_variables_without_sonderzeichen.append(var)
    sonderzeichen_score = sonderzeichen_score/count
    all_variables_without_sonderzeichen.extend(new)

    import enchant
    e = enchant.Dict("en_US")
    d = enchant.Dict("de_DE")
    for word in all_variables_without_sonderzeichen:
        if word == "":
            continue
        if not e.check(word) and not d.check(word):
            word_score += 1
    word_score = word_score/len(all_variables_without_sonderzeichen)
    interpretability_score = ((1-sonderzeichen_score)*x + (1-word_score)*y) + (1/len(decision_rule)*z)
    return interpretability_score
def eff_complexity(df, og_result, new_rule):
    hamming_distance = 0
    for rule in new_rule:
        df["hamming_distance"] = rule == og_result
        distance = df["hamming_distance"].value_counts()/df["hamming_distance"].size
        if distance[False]:
            distance = df["hamming_distance"].value_counts() / df["hamming_distance"].size
            hamming_distance += distance[False]


def calculate_interpretability_and_parsimony_weighted(weights_int, weights_par):
    xint,yint,zint = weights_int
    xpar,ypar,zpar = weights_par
    interpretability_list = []
    parsimony_list = []
    #print(calculate_interpretability_new("IF temperature.count(>=26.0)>=4.0 THEN 'Discard Goods'", ["temperature"], ["temperature.count(>=26.0)"]))
    pars_score = calculate_parsimony_new("IF temperature.count(>=26.0)>=4.0 THEN 'Discard Goods'", ["temperature"], ["temperature.count(>=26.0)"], xpar,ypar,zpar)
    int_score = calculate_interpretability_new("IF temperature.count(>=26.0)>=4.0 THEN 'Discard Goods'", ["temperature"], ["temperature.count(>=26.0)"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule = "IF temperature_intervall1_max > 25.5 AND temperature_intervall2_max > 25.5 AND temperature_intervall4_max > 25.5 THEN 'Discard Goods'"
    pars_score =calculate_parsimony_new(rule, ["temperature"],["temperature_intervall1_max", "temperature_intervall2_max","temperature_intervall4_max"],xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["temperature"],["temperature_intervall1_max", "temperature_intervall2_max","temperature_intervall4_max"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)


    rule= "IF temperature__quantile__q_0.8 > 25.90 AND temperature__change_quantiles__f_agg_\"var\"__isabs_True__qh_1.0__ql_0.6 <= 27.67 THEN \"Discard Goods\""
    pars_score =calculate_parsimony_new(rule, ["temperature"],["temperature__quantile__q_0.8", "temperature__change_quantiles__f_agg_\"var\"__isabs_True__qh_1.0__ql_0.6"], xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["temperature"],["temperature__quantile__q_0.8", "temperature__change_quantiles__f_agg_\"var\"__isabs_True__qh_1.0__ql_0.6"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)


    rule="IF measurement1>9.5 AND measurement1<=20.0 AND measurement2<=70.5 AND measurement0>19.5 AND measurement0<= 80.5 AND measurement2>29.5 THEN \"Put in OK pile\""
    pars_score =calculate_parsimony_new(rule, ["measurements"],["measurement1", "measurement2", "measurement0"], xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["measurements"],["measurement1", "measurement2", "measurement0"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule="IF tolerance2>measurement1=false AND measurement1<=tolerance2 = true AND tolerance2>=measurement2 = true AND measurement0<tolerance2 = false AND tolerance2>=measurement0 = true AND tolerance2>4>measurement2 = false THEN \"Put in OK pile\""
    pars_score =calculate_parsimony_new(rule, ["measurement", "tolerance"],["tolerance2>measurement1", "measurement1<=tolerance2", "tolerance2>=measurement2", "measurement0<tolerance2", "tolerance2>=measurement0", "tolerance2>measurement2"],xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["measurement", "tolerance"],["tolerance2>measurement1", "measurement1<=tolerance2", "tolerance2>=measurement2", "measurement0<tolerance2", "tolerance2>=measurement0", "tolerance2>measurement2"],xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule="IF measurement1<=tolerance0 AND measurement1<=tolerance2 AND measurement2<=70 AND measurement0<tolerance1 AND measurement0>=22 AND measurement2 >=30 THEN \"Put in OK pile\""
    pars_score =calculate_parsimony_new(rule, ["measurement", "tolerance"],["measurement1<=tolerance0", "measurement1<=tolerance2", "measurement2<=70", "measurement0<tolerance1", "measurement0>=22", "measurement2"],xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["measurement", "tolerance"],["measurement1<=tolerance0", "measurement1<=tolerance2", "measurement2<=70", "measurement0<tolerance1", "measurement0>=22", "measurement2"],xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule="IF diameter_intervall2_percentchange > 0.16 THEN \"Discard Goods\""
    pars_score =calculate_parsimony_new(rule, ["diameter"],["diameter_intervall2_percentchange"],xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["diameter"],["diameter_intervall2_percentchange"],xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule="IF casename<= 2242.5 AND casename<= 2179 AND casename<= 1932.5 AND diameter.last<= 27.25 THEN \"Discard Goods\""
    pars_score =calculate_parsimony_new(rule, ["casename", "diameter"],["casename", "diameter.last"], xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["casename", "diameter"],["casename", "diameter.last"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    rule = "IF diameter.count(>=38.28)>=3.0 AND diameter.count(>=39.78)>=4.0 AND diameter.count(>=39.8)>=4.0 == False THEN 'Discard Goods'"
    #print(calculate_interpretability_new(rule, ["diameter"],["diameter.count(>=38.28)", "diameter.count(>=39.78)","diameter.count(>=39.8)"]))
    pars_score =calculate_parsimony_new(rule, ["diameter"],["diameter.count(>=38.28)", "diameter.count(>=39.78)","diameter.count(>=39.8)"],xpar,ypar,zpar)
    int_score = calculate_interpretability_new(rule, ["diameter"],["diameter.count(>=38.28)", "diameter.count(>=39.78)","diameter.count(>=39.8)"], xint,yint,zint)
    interpretability_list.append(int_score)
    parsimony_list.append(pars_score)

    return interpretability_list, parsimony_list

def visualize(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    plt.figure(figsize=(30, 18))

    user_study = ["Understandability User Study", "Interpretability User Study", "Relevancy User Study", "Believability User Study", "Consistent User Study", "Concise User Study", "Complete User Study"]
    corr = df.corr()[user_study]#.sort_values(by=user_study, ascending=False)


    corr = corr.drop(user_study, axis=0, errors='raise')
    ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, annot_kws={"size":35} ,cmap='gray_r', cbar=False)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=35, rotation=0)
    xlabelsnew = [x.replace("User Study", "") for x in user_study]
    ax.set_xticklabels(labels=xlabelsnew, rotation=20, rotation_mode='anchor', ha='right', fontsize=35)
    plt.show()

def optimize_weights_interpretability(df):
    all_weights = generate_xyz_values()
    max_corr = 0
    optimal_weights = []
    n = 0

    for x,y,z in all_weights:
        interpretability_list, parsimony_list = calculate_interpretability_and_parsimony_weighted([x,y,z], [x,y,z])
        df["Interpretability_Weighted"] = interpretability_list
        act_corr = df['Interpretability Avg'].corr(df['Interpretability_Weighted'], method="pearson")
        n += 1
        if abs(act_corr) > abs(max_corr):
            optimal_weights = [x,y,z]
            max_corr = act_corr

    print("Maximum correlation Int: ", max_corr, optimal_weights)
    return optimal_weights

def optimize_weights_parsimony(df):
    all_weights = generate_xyz_values()
    max_corr = 0
    optimal_weights = []
    n = 0

    for x,y,z in all_weights:
        interpretability_list, parsimony_list = calculate_interpretability_and_parsimony_weighted([x,y,z],[x,y,z])
        df["Parsimony_Weighted"] = parsimony_list
        act_corr = df['Understandability Avg'].corr(df['Parsimony_Weighted'], method="pearson")
        n += 1
        if abs(act_corr) > abs(max_corr):
            optimal_weights = [x,y,z]
            max_corr = act_corr

    print("Maximum correlation Par: ", max_corr, optimal_weights)
    return optimal_weights

columns = ["Recall", "Precision","Useful","Accuracy","Parsimony", "Parsimony_New", "Interpretability", "Interpretability_new", "Interpretability_Scaled", "Combination_Scaled", "Combination", "Understandablity Median", "Understandablity Mode", "Understandability Avg", "Interpretability Median", "Interpretability Mode", "Interpretability Avg", "Concise Median", "Concise Mode", "Concise Avg", "Complete Median", "Complete Mode", "Complete Avg", "Consistent Median", "Consistent Mode", "Consistent Avg", "Useful Median", "Useful Mode", "Useful Avg", "Credible Median", "Credible Mode", "Credible Avg","Completeness","Effective Complexity" ,"Remine Consistency"]
data = [[100,100,"High",1,2,0.99,3,10,0.55,0.77,5.99,4.5,5,3.8,5,5,4, 5,5,4.7,4.5,5,4.3,5,5,4.5,5,5,4.5,5,5,4.5, 100,1.00,1],[69,71,"Medium",0.69,9,0.44,9,15,0.52,0.48,18.44,5,5,4.8,5,5,4.6, 4,5,3.9,3,5,3.2,5,5,4.2,3.5,2,3.4,3.5,3,3.7,65,0.50,0.2255639098], [99,99,"Low",1,4,0.825,12,40,0.43,0.63,16.825,1,1,1.6,1,1,1.3, 2.5,4,2.7,3,3,3.3,3,3,2.9,3,3,2.5,3,3,2.6,100,0.21,1] ,[100,100,"High",1,21,0.55,9,6,0.64,0.60,30.55,4.5,4,4.5,5,5,4.5, 3,3,3.22,5,5,4.3,5,5,4.9,4.5,5,4.2,5,5,4.4,100,0.08,1], [100,100,"Medium",1,38,0.275,19,37,0.60,0.44,57.275,2,1,2.4,2.5,3,2.4,2,1,2.4,3,2,2.9,2,1,2.5,2.5,1,2.5,3,3,2.44,100,0.08,0.7131] , [98,96,"Low",0.99,20,0.47,18,28,0.60,0.54,38.47,3.5,4,3.3,3,3,3.3,2.5,3,2.7,2.5,2,3.1,3,3,2.7,3,3,3.2,3,2,3,100,0.08,1], [91,92,"High",.91,3,0.99,4,6,0.41,0.70,7.99,5,5,4.6,4,4,4.2,5,5,4.4,4,4,3.5,5,5,4.8,4.5,5,3.9,4.5,5,3.9,100,1.00,1],[37,64,"Low",0.45,11,0.74,3,3,0.70,0.72,14.74,3.5,3,3.5,2.5,2,3,1.5,1,2.1,2,2,2.7,3,5,3.1,2.5,3,2.4,2.5,2,3, 33,0.29,0], [1,2,"Medium",0.09,9,0.44,9,32,0.53,0.49,18.44,3.5,3,3.6,3,3,3.2,2,2,2.1,3.5,3,3.7,2.5,2,2.8,2.5,3,2.5,2.5,3,2.5,0,0.00,0.6031746032] ]
df = pd.DataFrame(data= data, columns=columns)
weights_int = optimize_weights_interpretability(df)
weights_par = optimize_weights_parsimony(df)
#weights_par = [0.0, 0.8, 0.2]
#weights_int = [0.6,0.2,0.2]
interpretability_list, parsimony_list = calculate_interpretability_and_parsimony_weighted(weights_int, weights_par)
df["Interpretability_Weighted"] = interpretability_list
df["Parsimony_Weighted"] = parsimony_list
df["Completeness New"] = 1
df["Parsimony_Interpretability_Combined_Weighted"] = [x*0.15+y*0.85 for x, y in zip(parsimony_list, interpretability_list)]
df["F1_Score"] = 2*(df["Recall"]/100*df["Precision"]/100)/(df["Recall"]/100+df["Precision"]/100)
df.to_csv("experiments.csv")

df = pd.read_csv("experiments.csv")
pd.set_option('display.max_colwidth', None)
df_wanted = df[["F1_Score","Completeness New", "Effective Complexity","Accuracy","Interpretability_Weighted",  "Parsimony_Weighted", "Parsimony_Interpretability_Combined_Weighted",  "Remine Consistency New",  "Understandability Avg", "Interpretability Avg", "Useful Avg", "Credible Avg", "Consistent Avg", "Concise Avg", "Complete Avg"]]
df_wanted.rename(columns={'Interpretability_Weighted': 'Interpretability', 'Completeness New': 'Completeness', 'Parsimony_Weighted': 'Parsimony','Parsimony_Interpretability_Combined_Weighted': 'Interpretability\n&Parsimony\n(I&P)', "Remine Consistency New": 'Remine\nConsistency\n(RC)', "Effective Complexity": "Effective\nComplexity\n(EC)", "F1_Score":"F1 Score"}, inplace=True)
df_wanted.rename(columns={'Useful Avg': 'Relevancy Avg', 'Credible Avg': 'Believability Avg'}, inplace=True)
df_wanted.columns = df_wanted.columns.str.replace(' Avg', ' User Study')
print(df_wanted.columns)
visualize(df_wanted)
