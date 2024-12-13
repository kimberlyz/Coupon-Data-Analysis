# Coupon Data Analysis

In this assignment, our task is to answer the question “Will a customer accept the coupon?” - essentially giving a recommendation on how to optimize a coupon campaign to yield the best coupon acceptance results. The details of the coupon campaign involve delivering digital coupons to drivers in various redemption categories - a bar, a coffee shop, a cheaper restaurant (under $20), and a more expensive restaurant ($20 - $50). 

You can see the full code for the data cleaning and analysis steps in the Jupyter notebook CouponRecommendation.ipynb. This ReadMe will contain a summary of the code and data analysis process in that file. 

We were given the Problem Statement below.

## Problem Statement
**Context**

Imagine driving through town and a coupon is delivered to your cell phone for a restaurant near where you are driving. Would you accept that coupon and take a short detour to the restaurant? Would you accept the coupon but use it on a subsequent trip? Would you ignore the coupon entirely? What if the coupon was for a bar instead of a restaurant? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car? What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?

Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?

**Overview**

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those that did not.

**Data**

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. There are three possible answers people can choose from:

1. “Right away”
2. “Later, before the coupon expires”
3. “No, I do not want the coupon”
   
The first two responses are labeled as “Y = 1,” and the third is labeled as “Y = 0.” There are five different types of coupons: Less expensive restaurants (under $20), coffee houses, carryout and takeaway, bars, and more expensive restaurants ($20–$50).

Keep in mind that these values mentioned below are average values.

The attributes of this data set include:
1. User attributes
    -  Gender: male, female
    -  Age: below 21, 21 to 25, 26 to 30, etc.
    -  Marital Status: single, married partner, unmarried partner, or widowed
    -  Number of children: 0, 1, or more than 1
    -  Education: high school, bachelors degree, associates degree, or graduate degree
    -  Occupation: architecture & engineering, business & financial, etc.
    -  Annual income: less than $12500, $12500 - $24999, $25000 - $37499, etc.
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
    than 8
    -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
    greater than 8
    -  Number of times that he/she eats at a restaurant with average expense less than $20 per
    person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    

2. Contextual attributes
    - Driving destination: home, work, or no urgent destination
    - Location of user, coupon and destination: we provide a map to show the geographical
    location of the user, destination, and the venue, and we mark the distance between each
    two places with time of driving. The user can see whether the venue is in the same
    direction as the destination.
    - Weather: sunny, rainy, or snowy
    - Temperature: 30F, 55F, or 80F
    - Time: 10AM, 2PM, or 6PM
    - Passenger: alone, partner, kid(s), or friend(s)


3. Coupon attributes
    - time before it expires: 2 hours or one day

## Data Cleaning
Before I start analyzing the data, I want to perform some data cleaning. 

The data cleaning steps performed were the following.

1. Removed duplicates
2. Removed car column
3. Removed rows with NaN in them
4. Standardize column names and fix typos in names

After all the steps above, I end up with 12,007 rows x 25 columns instead of the original 12,684 rows × 26 columns.

Here are the reasonings behind why I could do each step.

1. Removed duplicates: There were 74 duplicate rows out of 12,684 total rows, which accounts for only .58% of the data. I am hypothesizing these are accidentally re-submitted / counted values for the same person because it is unlikely that the same exact factors across all the columns are legitimate separate inputs that apply to different people. Thus, let's remove these duplicate values.

2. Removed car column: From data.info(), I can see there is a lot of missing data in the car column. Only 108 out of 12610 rows have car data, which is equivalent to only 0.86% of the data. Thus, let's just drop this column since there's not enough information to yield valuable insights.

3. Removed rows with NaN in them: Because the percentage of rows containing an NaN value is < 5% of all the data, I think it's okay to drop those rows as there's still enough rows left to analyze.

4. Standardize column names and fix typos in names: To make it easier to remember what the column names are when typing them out, let's fix any typos and standardize the column names into snakecase.

## Data Analysis / Manipulation
**Overview of Data**

Here is a preview of some of the actual values of the dataset: 
![data example](https://github.com/user-attachments/assets/ac414f8b-c1c4-43be-b620-f943167f34fd)

I got a general overview of the data by doing:
1. `df.describe()` - gets stats like the count, mean, std, quartiles, etc. of all the quantitative columns
2. `df.columns` - took a look at what columns existed
3. `df[col_name].value_counts()` - take a look at what values exist per column
4. Some plots

We can see out of the entire cleaned dataset, 56.85% of people accepted the coupon.

<img src="https://github.com/user-attachments/assets/3ae88072-fb01-43b2-ae32-1e434962b8e5" width=300/>

We can also see that the most popular coupons were Carry Out and Restaurants(<20), while the least popular coupon was Bar. 

<img src="https://github.com/user-attachments/assets/96111a24-558a-4384-8e38-a0fab1e7c0fc" width=450/>

**One Hot Encoding and Correlation Analysis**

Once I got a general data overview, I then wanted to explore correlations with one hot encoding. 
I only wanted to one hot encode categorical columns if they didn't have too many unique values - that way the dimensionality of the data doesn't increase too greatly. I ended up one hot encoding columns with <= 5 unique values, and ended up adding 59 more columns. After one hot encoding I ended up with 71 columns, which still seemed manageable.

I saw a natural division in the data between correlations above 0.1 and below -0.1 as follows.
| Column Name | Correlation |
|-| -|
| destination_No Urgent Place     |0.130559|
| passenger_Friend(s)             |0.129314|
| weather_Sunny                   |0.100143|
| coupon_Carry out & Take away    |0.161511|
| coupon_Restaurant(<20)          |0.151708|
| expiration_1d                   |0.131872|

| Column Name | Correlation |
|-| -|
|to_coupon_GEQ_25_min   |-0.105185|
|passenger_Alone        |-0.100726|
|coupon_Bar             |-0.137309|
|coupon_Coffee House    |-0.100120|
|expiration_2h          |-0.131872|
|coffee_never           |-0.129865|

Thus, people are more likely to accept the coupon if one of the column names is true in the positive correlations table. And people are less likely to accept the coupon if one of the column names is true in the negative correlations table.

**Percentage Acceptance Rate Per Column**

I then was curious about the highest percentage coupon acceptance rates per column value. For example, choosing a column value like "people who received a bar coupon", and seeing that out of all "people who received a bar coupon", how many of them ended up accepting the coupon.

I then got the following with the highest acceptance rates. 

| Column Value | Percentage Coupon Accpet |
|-| -|
| coupon_Carry out & Take away| 73.58%|
| coupon_Restaurant(<20)| 70.98%|
| passenger_Friend(s)| 67.60%|
| time_2PM| 66.07%|
| restaurant_20_to_50_gt8| 66.29%|
| restaurant_20_to_50_4~8| 65.35%|
| coffee_1~3| 65.11%|

## Takeaways / Recommendations
In general, the most effective coupons were:
1. Carry out coupons: 73.58% acceptance
2. Cheaper restaurant (<$20) coupons: 70.98% acceptance

People were also more likely to use the coupon if one of the following applied:
1. had a passenger who was a friend with them
2. their destination was no urgent place
3. the coupon expiration date was 1 day.

People were less likely to use the coupon if one of the following applied: 
1. it took greater than or equal 25 minutes to get to the coupon destination
2. the coupon expiration date was in 2 hours
3. the coupon was a Bar coupon
4. they never go to a coffee shop each month

So, if people were to run a more effective driver coupon delivery campaign, the recommendation would be to focus on giving carry out coupons and cheaper restaurants (<$20) coupons with a 1 day expiration time. It would also be ideal if the coupon was for a restaurant < 25 minutes away. While it would be good to also deliver the coupon if the driver had a friend with them and the driver did not have an urgent destination to get to, this is real-time information that would be difficult for the ad company to have in the moment and thus they would be unable to target coupon delivery this way.
