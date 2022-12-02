# This is a sample Python script.
import pandas as pd
import numpy as np
import plotly.express as px
import datetime as dt
import pycountry
import requests



#import matplotlib
#import oracle.connector





# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#See https://www.kaggle.com/code/nemataloush/aviation-disasters-analysis for source


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_country_code(country):
    """
    This function returns the ISO Code of a country.

    Argument:
    country -- A string value holds a country name

    Returns:
    result -- The ISO Alpha-3 code of 'country' argument.
    """
    try:
        result = pycountry.countries.search_fuzzy(country)
    except Exception:
        return np.nan
    else:
        return result[0].alpha_3


# So what precentage for each country ?
def precentage(integer):
    return integer * 100 / num_accidents


# So what precentage for each operator ?
def precentage(integer):
    return round(integer * 100 / num_accidents, 2)


# Defining 2 functions to add 2 colomns about the accident cause and result
def cause_accident(value):
    if value.find('A')>=0:
        return 'Accident'
    if value.find('I')>=0:
        return 'Incident'
    if value.find('H')>=0:
        return 'Hijacking'
    if value.find('C')>=0:
        return 'Criminal occurrence'
    if value.find('O')>=0:
        return 'Other occurrence'
    if value.find('U')>=0:
        return 'Unknown'

def result_accident(value):
    if value.find('1')>=0:
        return 'Hull Loss'
    if value.find('2')>=0:
        return 'Repairable Damage'


#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#request = requests.get(' http://api.open-notify.org/')
#print(request.status_code)


data = pd.read_csv("aviation_accidents in countries - aviation_accidents.csv")
num_accidents = data.shape[0] # saving the count of total accidents
print(data.head())


# How many null values do we have?
nulls_count_precentage={col_name : (data[col_name].isna().sum(),
                                str( round( 100 * data[col_name].isna().sum() / data.shape[0] ,2))+" %")
                                for col_name in data.columns}
#print(nulls_count_precentage)

#print(data.nunique())

iso_map = {country: get_country_code(country) for country in data["Country"].unique()}

data["country_code"] = data["Country"].map(iso_map)

countries_df = data.groupby(['Country','country_code']).size().sort_values(ascending=[False])\
         .to_frame().reset_index().rename(columns={0: 'count_accidents'})

# Reformating the dataframe keeping the details of only the top 15 country and combine the other countries in one row
# 'other'
countries_df_part=countries_df[:15]
accidents_other = countries_df['count_accidents'][15:].sum()
df2 = pd.DataFrame([['other','other', accidents_other]], columns=['Country','country_code','count_accidents'])
countries_df_part=countries_df_part.append(df2)

fig = px.pie(countries_df_part,
            values='count_accidents',
            names='Country',
            title='15 countries where the most accidents happend')
fig.update_traces( textinfo='value+label',textfont_size=10)
fig.show()





output_list = list(map(precentage, countries_df_part['count_accidents']))

print(output_list)


fig = px.scatter_geo(countries_df, locations="country_code",
                     color="Country", # which column to use to set the color of markers
                     hover_name="Country", # column added to hover information
                     size="count_accidents", # size of markers
                     projection="natural earth")
fig.show()

# Keeping the top 20 countries where the most accidents happened
countries_most_accidents = countries_df['Country'][:20]

data["operator"]=data["operator"].str.split(',').str[0]

operators = data.groupby(['operator']).size()\
    .sort_values(ascending=[False]).to_frame()\
        .reset_index().rename(columns= {0: 'count_accidents'})

# reformating the dataframe, keeping the details of only th top 10 operator
operators_part=operators[:10]
accidents_other = operators['count_accidents'][10:].sum()
df2 = pd.DataFrame([['other', accidents_other]], columns=['operator','count_accidents'])
operators_part=operators_part.append(df2)

fig = px.pie(operators_part,
            values='count_accidents',
            names='operator',
            title='10 operators which suffered the most number of accidents')
fig.update_traces( textinfo='value+label',textfont_size=10)
fig.show()




# So what precentage for each operator ?
output_list = list(map(precentage, operators_part['count_accidents']))

print(output_list)


# Keeping the top 20 operators that had the most accidents
operators_most_accidents = operators['operator'][:20]



countries_operators_df = data.groupby(['Country','country_code','operator']).size().sort_values(ascending=[False])\
    .to_frame().reset_index()\
    .rename(columns= {0: 'count_accidents'})


# In each country, what operators made the most accidents
fig = px.treemap(countries_operators_df, path=[px.Constant('world'), 'Country', 'operator'], values='count_accidents',
                   hover_data=['country_code'])
fig.update_traces(root_color="lightgrey")
fig.show()


# Keeping the bair of country-operator where both the counrtry and the operator are from the top 20 of count of accidents.
countries_operators_part= countries_operators_df[countries_operators_df['Country'].isin(countries_most_accidents)]
countries_operators_part= countries_operators_part[countries_operators_df['operator'].isin(operators_most_accidents)]


# In each operator (of the top 20),  in which countries (of the top 20) those operators made the most accidents
fig = px.treemap(countries_operators_part, path=[px.Constant('world'), 'operator','Country'], values='count_accidents',
                   hover_data=['country_code'])
fig.update_traces(root_color="lightgrey")
fig.show()


# As we have seen already, there are some null values in the category colomn,
# so we will drop them first.
categories_filtered=data.dropna(subset=['category'])


## Adding 2 columns representing the cause and result of the data individually
causes = categories_filtered['category'].map(cause_accident)
results = categories_filtered['category'].map(result_accident)
categories_filtered = pd.concat([categories_filtered, causes, results], axis=1, join="inner")
categories_filtered.columns = ['Country', 'date', 'Air-craft type', 'registration name/mark',
       'operator', 'fatilites', 'location', 'category', 'country_code',
       'Accident_cause', 'Accident_result']


# Finding the pair of result-cause combination
result_cause = categories_filtered.groupby(
                                ['Accident_cause','Accident_result'])  \
                                .size().sort_values(ascending=False).  \
                                to_frame().reset_index().  \
                                rename(columns={0: 'count_accidents'})



# the accident where the result where hull loss
hull_loss_df = result_cause[result_cause['Accident_result']=='Hull Loss']
# the accident where the result where Repairable Damage
repairable_df = result_cause[result_cause['Accident_result']=='Repairable Damage']
#
result_by_cause_df = pd.merge(hull_loss_df, repairable_df, how='outer', on = 'Accident_cause').fillna(0)
result_by_cause_df[:2]


# Defining a list for the sake of visualization
to_draw= [result_by_cause_df["count_accidents_x"].to_list()\
         ,result_by_cause_df["count_accidents_y"].to_list()]



fig = px.imshow(to_draw,
                labels=dict(x="Cause of Accident", y="Result of Accident", color="Accidents"),
                x= result_by_cause_df["Accident_cause"],
                y=["Hull Loss","Repairable Damage"],
                text_auto = True
               )
fig.update_xaxes(side="top" )
fig.show()


# Finding the precentage of each type
def precentage(integer):
    return [round(i*100/num_accidents,2) for i in integer]
output_list = list(map(precentage, to_draw))
print(output_list)

print("Precentage of Accidents that caused Hull Loss",round(sum (output_list[0]),2))
print("Precentage of Accidents that caused Repairable damage",round(sum(output_list[1]),2))

#Studying the accidents regarding the operator and cause of accidents
category_operator= categories_filtered.groupby(
                                ['operator','Accident_result', 'Accident_cause'])\
                                .size().sort_values(ascending=False).to_frame()\
                                .reset_index().rename(columns={0: 'count_accidents'})
category_operator= category_operator[category_operator['operator'].isin(operators_most_accidents[:4])]

fig = px.sunburst(category_operator, path=['Accident_result','Accident_cause', 'operator'], values='count_accidents',
                  color='count_accidents')
fig.show()


country_category=categories_filtered.groupby(
                      ['Country', 'Accident_result','Accident_cause']
                      ).size().sort_values(ascending=False).to_frame(
                      ).reset_index().rename(columns={0: 'count_accidents'})

country_category= country_category[country_category['Country'].isin(countries_most_accidents[:4])]

fig = px.bar(country_category, x="Country", y="count_accidents", color="Accident_cause",
             pattern_shape="Accident_result", pattern_shape_sequence=["x", "."])

fig.show()

data["Air-craft type"]=data["Air-craft type"].str.split(' ').str[0]
categories_filtered["Air-craft type"]=categories_filtered["Air-craft type"].str.split(' ').str[0]

air_crafts = data.groupby(['Air-craft type']).size()\
    .sort_values(ascending=[False]).to_frame().reset_index()\
        .rename(columns= {0: 'count_accidents'})

air_craft_part=air_crafts[:10]
accidents_other = air_crafts['count_accidents'][10:].sum()
df2 = pd.DataFrame([['other', accidents_other]], columns=['Air-craft type','count_accidents'])
air_craft_part=air_craft_part.append(df2)


fig = px.pie(air_craft_part,
            values='count_accidents',
            names='Air-craft type',
            title='10 Air crafts which suffered the most number of accidents')
fig.update_traces( textinfo='value+label',textfont_size=10)
fig.show()


# So what precentage for each aircraft ?
def precentage(integer):
    return round(integer*100/num_accidents,2)
output_list = list(map(precentage, air_craft_part['count_accidents'][:12]))
print(output_list)
print(sum(output_list))


# Keeping the top 4 aircrafts with most accidents
air_craft_top = air_craft_part[:4]


air_craft_category=categories_filtered.groupby(
                      ['Air-craft type', 'Accident_result','Accident_cause']
                      ).size().sort_values(ascending=False).to_frame(
                      ).reset_index().rename(columns={0: 'count_accidents'})


air_craft_category_filtered= air_craft_category[air_craft_category['Air-craft type'].isin(air_craft_top['Air-craft type'])]

print(air_craft_top)

fig = px.bar(air_craft_category_filtered, x="Air-craft type",\
     y="count_accidents", color="Accident_cause",
 pattern_shape="Accident_result", pattern_shape_sequence=[ "x","."])
fig.show()

#Studying the data regarding the date

#### Dropping rows with unknow dates
dates_filtered = data.drop(data[(data.date == 'date unk.') ].index)
#### Dropping rows with null dates
dates_filtered=dates_filtered.dropna(subset=['date'])

#### Adding a year & decade colomns
dates_filtered['year'] = dates_filtered['date'].str[-4:].str.lower()
dates_filtered['decade'] = dates_filtered['year'].astype(int).round(-1)


#Finding the number of accidents for each country in each decade
decade_country= dates_filtered.groupby(['Country','country_code',\
    'decade']).size().to_frame()\
    .reset_index().rename(columns={0: 'accidents'})


fig = px.line(decade_country, x="decade", y="accidents", color="Country", line_group="Country", hover_name="Country",
        line_shape="spline", render_mode="svg")
fig.show()

# Anthore type of representation

df = px.data.medals_long()
fig = px.bar(decade_country, x="decade", y="accidents", color="Country")
fig.show()

# Countries to be shown
decade_show_country=decade_country[decade_country['Country'].isin(countries_most_accidents[:14])]
#countries to be combined
#countries to be combined
decade_other_country =decade_country[~decade_country['Country'].isin(countries_most_accidents[:14])]
decade_other_country = decade_other_country.groupby('decade').size().to_frame()\
    .reset_index().rename(columns={0: 'accidents'})
decade_other_country.insert(0, 'Country', 'Other')
decade_other_country.insert(1, 'country_code', 'Other')
decade_other_country[:2]

# the final dataframe to be plotted.
decade_country_finale = pd.concat([decade_show_country,decade_other_country])
fig = px.bar(decade_country_finale, x="decade", y="accidents", color="Country")
fig.show()








