#!/usr/bin/env python
# coding: utf-8

#  # Data Analysis of Kpop Industry

# ![kpop-photo.jpg](attachment:kpop-photo.jpg)

# In this analysis, we will conduct an Exploratory Data Analysis (EDA) on a dataset that contains details about K-pop idols. The dataset consists of various information including stage names, full names, birth dates, group affiliations, debut dates, company associations, nationalities, heights, weights, birthplaces, and genders of the idols. The main objective of this EDA is to explore the data and identify any noteworthy trends, patterns, or correlations within the dataset. By doing so, we aim to gain valuable insights into the K-pop industry, how they works and what is their idol choosing criteria. 

# ## Lets begin with the analysis 

# In[1]:


#importing necessary libraries

import pandas as pd #data processing
import numpy as np #linear algebra
import matplotlib as plt #for creating plot & visualization
import seaborn as sns #provides additional functionality for statistical data

#note: downgrading numpy as numpy.bool is deprecated

import pandas as pd
import plotly.express as px #interactive plots

#control of warning messages & prevent them from being shown in the output
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime # work with dates and times


# In[2]:


#inline display of matplotlib plots directly in the notebook.
#for high resolution graphics


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")


# In[4]:


#pip install plotly if its not available in your notebook.


# In[5]:


#import & read the data
data = pd.read_csv("C:/Users/HP/Downloads/kpopidolsv3.csv")


# In[6]:


#print the data to further analysis
print(data)


# ## Data cleaning & Data Prep 

# In[7]:


#cheking missing ratio from particular column

data_na = (data.isnull().sum() / len(data)) * 100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :data_na})
missing_data.head(20)


# In[8]:


# check duplicate rows in the data
duplicate_rows_data = data[data.duplicated()]
print("number of duplicate rows: ", duplicate_rows_data.shape)


# In[9]:


# Loop through each column and count the number of distinct values
for column in data.columns:
    num_distinct_values = len(data[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")


# In[10]:


data['Debut'] = data['Debut'].replace('0/01/1900', pd.NA)


# In[11]:


data['Date of Birth'] = pd.to_datetime(data['Date of Birth'], format='%d/%m/%Y')
data['Debut'] = pd.to_datetime(data['Debut'], format='%d/%m/%Y')

data['age'] = (datetime.now() - data['Date of Birth']).astype('<m8[Y]')
data['Debut Age'] = (data['Debut'] - data['Date of Birth']).astype('<m8[Y]')


# In[12]:


# Define a function to create zodiac sign
def get_zodiac_sign(date):
    if (date.month == 1 and date.day >= 20) or (date.month == 2 and date.day <= 18):
        return 'Aquarius'
    elif (date.month == 2 and date.day >= 19) or (date.month == 3 and date.day <= 20):
        return 'Pisces'
    elif (date.month == 3 and date.day >= 21) or (date.month == 4 and date.day <= 19):
        return 'Aries'
    elif (date.month == 4 and date.day >= 20) or (date.month == 5 and date.day <= 20):
        return 'Taurus'
    elif (date.month == 5 and date.day >= 21) or (date.month == 6 and date.day <= 20):
        return 'Gemini'
    elif (date.month == 6 and date.day >= 21) or (date.month == 7 and date.day <= 22):
        return 'Cancer'
    elif (date.month == 7 and date.day >= 23) or (date.month == 8 and date.day <= 22):
        return 'Leo'
    elif (date.month == 8 and date.day >= 23) or (date.month == 9 and date.day <= 22):
        return 'Virgo'
    elif (date.month == 9 and date.day >= 23) or (date.month == 10 and date.day <= 22):
        return 'Libra'
    elif (date.month == 10 and date.day >= 23) or (date.month == 11 and date.day <= 21):
        return 'Scorpio'
    elif (date.month == 11 and date.day >= 22) or (date.month == 12 and date.day <= 21):
        return 'Sagittarius'
    else:
        return 'Capricorn'

# Apply the function to 'Date of Birth' column to get 'Zodiac' column
data['Zodiac'] = data['Date of Birth'].apply(get_zodiac_sign)


# In[13]:


data = data.drop(['Korean Name','K Stage Name','Second Country'], axis=1)


# In[14]:


data.head()


# ## Does Height Matter of Kpop idols?

# In[15]:


#!pip install --upgrade seaborn


# ![BMI.png](attachment:BMI.png)
# 
# BMI= weight in kg/ height square in meters

# In[16]:


# pip install --upgrade matplotlib


# In[17]:


#pip show matplotlib


# In[18]:


# pip install --force-reinstall matplotlib


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


# Separate data for males and females
male_data = data[data['Gender'] == 'M']
female_data = data[data['Gender'] == 'F']

# Set color palette
sns.set_palette('pastel')

# Plotting distribution with average for males and females
sns.histplot(data=male_data, x='Height', kde=True, color='skyblue', label='Male')
sns.histplot(data=female_data, x='Height', kde=True, color='lightpink', label='Female')
plt.axvline(x=male_data['Height'].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Male Avg')
plt.axvline(x=female_data['Height'].mean(), color='red', linestyle='dashed', linewidth=1.5, label='Female Avg')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.title('Height Distribution - Males vs. Females')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

plt.axvline(x=male_data['Height'].mean(), color='blue', linestyle='dashed', linewidth=1.5, label='Male Avg')
plt.axvline(x=female_data['Height'].mean(), color='red', linestyle='dashed', linewidth=1.5, label='Female Avg')


# ## What body type is ideal for being an K-POP Idol?

# In[21]:


# Filter rows where both 'Height' and 'Weight' are not null
filtered_data = data.dropna(subset=['Height', 'Weight'])

# Calculate BMI
filtered_data['BMI'] = filtered_data['Weight'] / ((filtered_data['Height'] / 100) ** 2)

# Define BMI categories and corresponding colors
bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_category_colors = ['blue', 'green', 'orange', 'red']
bmi_category_ranges = [0, 18.5, 25, 30, np.inf]

# Determine BMI category for each individual
filtered_data['BMI Category'] = pd.cut(filtered_data['BMI'], bins=bmi_category_ranges, labels=bmi_categories)

# Convert BMI categories to numeric codes
filtered_data['BMI Category Code'] = pd.Categorical(filtered_data['BMI Category'], categories=bmi_categories).codes
# Separate data by gender
male_data = filtered_data[filtered_data['Gender'] == 'M']
female_data = filtered_data[filtered_data['Gender'] == 'F']

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Male BMI graph
male_scatter = ax1.scatter(male_data['Height'], male_data['Weight'], c=male_data['BMI Category Code'], cmap='coolwarm', edgecolors='black')
ax1.set_xlabel('Height (cm)')
ax1.set_ylabel('Weight (kg)')
ax1.set_title('Male BMI Distribution')

# Female BMI graph
female_scatter = ax2.scatter(female_data['Height'], female_data['Weight'], c=female_data['BMI Category Code'], cmap='coolwarm', edgecolors='black', vmin=0, vmax=len(bmi_categories)-1)
ax2.set_xlabel('Height (cm)')
ax2.set_ylabel('Weight (kg)')
ax2.set_title('Female BMI Distribution')

# Colorbar for male graph
cbar_male = plt.colorbar(male_scatter, ax=ax1, ticks=np.arange(len(bmi_categories)))
cbar_male.set_label('BMI Category (Male)')
cbar_male.set_ticklabels(bmi_categories)

# Colorbar for female graph
cbar_female = plt.colorbar(female_scatter, ax=ax2, ticks=np.arange(len(bmi_categories)))
cbar_female.set_label('BMI Category (Female)')
cbar_female.set_ticklabels(bmi_categories)

plt.tight_layout()
plt.show()


# Comment: The result appear above like the majority of K-pop stars are underweight

# ## Age At The Time Of Their Debut

# In[22]:


# Dealing with wring value
male_data = male_data[male_data['Debut Age'] >= 10]
female_data = female_data[female_data['Debut Age'] >= 10]

# Calculate average debut age for males and females
male_avg = male_data['Debut Age'].mean()
female_avg = female_data['Debut Age'].mean()

# Plotting the 'Debut Age' for males and females
plt.hist(male_data['Debut Age'].dropna(), bins=10, alpha=0.5, label='Male')
plt.hist(female_data['Debut Age'].dropna(), bins=10, alpha=0.5, label='Female')
plt.axvline(x=male_avg, color='blue', linestyle='dashed', linewidth=1, label='Male Average')
plt.axvline(x=female_avg, color='red', linestyle='dashed', linewidth=1, label='Female Average')
plt.xlabel('Debut Age')
plt.ylabel('Frequency')
plt.title('Debut Age Distribution - Male vs. Female')
plt.legend()
plt.show()


# ## Age Distribution

# In[23]:


# Calculate average age for males and females
male_avg_age = male_data['age'].mean()
female_avg_age = female_data['age'].mean()

# Plot age distribution for males
plt.hist(male_data['age'], bins=20, color='blue', alpha=0.5, label='Male')

# Plot age distribution for females
plt.hist(female_data['age'], bins=20, color='red', alpha=0.5, label='Female')

# Add average lines for males and females
plt.axvline(male_avg_age, color='blue', linestyle='--', linewidth=2, label='Male Avg Age')
plt.axvline(female_avg_age, color='red', linestyle='--', linewidth=2, label='Female Avg Age')

# Set plot labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Gender')
# Add a legend
plt.legend()

# Display the plot
plt.show()


# ## What nation are K-POP idols from?

# In[24]:


df =data.copy()
# Count the number of K-pop idols by birthplace (country)
birthplace_counts = df['Country'].value_counts().reset_index()
birthplace_counts.columns = ['Country', 'Count']

# Create a choropleth map using Plotly with a log scale for color
fig = px.choropleth(birthplace_counts,
                    locations='Country',
                    locationmode='country names',
                    color=np.log10(birthplace_counts['Count']),
                    hover_name='Country',
                    custom_data=[birthplace_counts['Count']],
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title='K-pop Idols by Birthplace (Country) - Log Scale',
                    labels={'color': 'Log10(Number of K-pop Idols)'},
                    projection='natural earth')

# Create a custom hover template
hovertemplate = "<b>%{hovertext}</b><br>Number of K-pop Idols: %{customdata[0]}<br>Log10(Number of K-pop Idols): %{z:.2f}<extra></extra>"

# Update the hover template using the update_traces() method
fig.update_traces(hovertemplate=hovertemplate)

fig.show()


# ## Top 10 Birthplaces of kpop idols

# In[25]:


# Filter the data for rows where the 'Country' column is 'South Korea'
south_korea_data = data[data['Country'] == 'South Korea']

# Count the occurrences of each birthplace and get the top 10
birthplace_counts = south_korea_data['Birthplace'].value_counts().head(10)

# Plot the top 10 birthplace counts
plt.bar(birthplace_counts.index, birthplace_counts.values)
plt.xlabel('Birthplace')
plt.ylabel('Count')
plt.title('Top 10 Birthplaces of Idols in South Korea')
plt.xticks(rotation=90)
plt.show()


# ## How frequently do idol debuts?

# In[26]:


# Convert the 'Debut' column to datetime
data['Debut'] = pd.to_datetime(data['Debut'])

# Group by year and count the number of debuts
debut_counts = data['Debut'].dt.year.value_counts().sort_index()

# Plot the idol debuts over time
plt.plot(debut_counts.index, debut_counts.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Debuts')
plt.title('Idol Debuts Over Time')
plt.show()


# ## Zodiac Sign

# In[27]:


# Define colors for each zodiac sign
zodiac_colors = {
    'Aquarius': '#1f77b4',
    'Pisces': '#ff7f0e',
    'Aries': '#2ca02c',
    'Taurus': '#d62728',
    'Gemini': '#9467bd',
    'Cancer': '#8c564b',
    'Leo': '#e377c2',
    'Virgo': '#7f7f7f',
    'Libra': '#bcbd22',
    'Scorpio': '#17becf',
    'Sagittarius': '#ff9896',
    'Capricorn': '#9edae5'
}

# Count the occurrences of each zodiac sign
zodiac_counts = data['Zodiac'].value_counts()
# Sort the zodiac counts by the zodiac sign order
sorted_zodiac_counts = zodiac_counts.loc[list(zodiac_colors.keys())]

# Plot the zodiac sign distribution with correlated colors
fig, ax = plt.subplots()
bars = ax.bar(sorted_zodiac_counts.index, sorted_zodiac_counts.values, color=[zodiac_colors[zodiac] for zodiac in 
                                                                              sorted_zodiac_counts.index])

# Add labels and percentages above each bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:,}', ha='center', va='bottom', fontsize=9)

# Customize the plot
plt.xlabel('Zodiac Sign')
plt.ylabel('Count')
plt.title('Distribution of Idols by Zodiac Sign')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()


# In[28]:


# to get the information about entire data
data.info()


# In[29]:


# to get unique group names i.e to avoid duplicate group names
data.Group.unique()


# In[30]:


# occurences of each group in this dataset
group_count = data['Group'].value_counts()
group_count = group_count[:10,]
group_count


# In[31]:


#to create horizontal bar plot with count values on x-axis & y-axis using seaborn(sns)
sns.barplot(y = group_count.index, x= group_count.values)


# In[32]:


# to check how many times top 10 companies occur in a dataset
company_count = df['Company'].value_counts()
company_count = company_count[:10,]


# In[33]:


# graphical representation of the company count
sns.barplot(x= company_count.values,y=company_count.index)


# In[ ]:





# # Conclusion
# Analyzing data in the K-pop industry based on idols' height, weight, country, debut time, and age can provide valuable insights into trends and patterns. Here are some possible conclusions that can be drawn from such analysis:
# 
# 1. Diversity in Nationalities: The K-pop industry has become increasingly global, with idols from various countries making their debut. The data analysis might reveal a diverse representation of nationalities, highlighting the international appeal of K-pop.
# 
# 2. Physical Appearance Standards: K-pop idols are often associated with strict beauty standards. By examining the height and weight data, it may be possible to identify the prevalent physical appearance ideals within the industry. However, it's important to note that individual differences and personal preferences also play a role.
# 
# 3. Age and Debut Time: Analyzing the age and debut time of idols can offer insights into the industry's preferences for certain age groups. It might indicate whether there is a preference for younger or older debuts and whether there are any specific trends regarding the timing of debuts.
# 
# 4. Comparing Height and Weight: By comparing the height and weight data of idols, it's possible to identify any prevalent patterns or trends within the industry. This analysis could reveal whether there is a preference for specific body types or if there is a wider range of body shapes and sizes among idols.
# 
# 5. Evolution Over Time: Analyzing data across different time periods can provide insights into how the industry has evolved. For example, it may reveal changes in physical appearance standards, the diversity of nationalities, or the age at which idols debut.
# 
# It's important to note that these conclusions are speculative and based on hypothetical data analysis. In reality, the K-pop industry is complex, and individual factors such as talent, personality, and marketability also play a significant role in an idol's success. Additionally, the industry is constantly evolving, and trends can change over time.

# In[ ]:




