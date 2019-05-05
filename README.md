# HR_turnover
Predicting turnover of employees using Random Forest Classifier. 

The datasets used here are not public datasets, they belong to a real international company, therefore are not uploaded here. 
The notebooks are for displaying the code only. I will update with a mock data set that is structurally identical to the real dataset later. 

## The Data 
The data belongs to an international company with over 20.000 employees working in more than 20 countries. One of the challenges was the differences in data collection methods in the different countries the company operates in. For example, the comparison of pay would be meaningless, as there are different policies/pay levels in each continent. However a lot of valuable information would be lost if I restricted the analysis to one continent, because this company had very low levels of turnover (around 1% in the last year), and I had to include all the data I had. 

The data included the last 13 months of info per employee. Not all employees had all 13 months, as the query did not bring up rows from months were there were no change to the employee info. Some of the "leavers" (employees who lef the company) also didn't have all the months, as they left before the last month I pulled from the server. 

The data:
* main: the data with basic demographics such as age, gender, seniority level, years spent in the company, nationality etc.
* Leavers: the employees who left the company in the last year. Only resignation, excluding retirment and death
* attend: average working hours entered per month
* absencence: average vacation hours entered per month
* travel: travel hours entered

The challenge was to merge these datasets without loosing detail of info as much as possible. I couldn't use data such as sickleave, motivation or pay due to GDPR, which would probably increase the performance of the model immediately. That pushed me to get creative with data engineering. I created the following 

* expat: if the nationality of the employee was different than the country they worked in 
* promotion: if the employee changed to a position that's higher in the corporate ladder
* country_count: number of countries an employee worked in. more diverse experience considered to improve motivation but also increase the employee's value, could lead to poaching. 

## The Model

I wanted to predict which employees would leave the company, so I needed a classifier. I narrowed it down to shallow classifiers as my data was very unbalanced (from around 20.000 employees, only 300 were leavers) and realtively small. 

I also knew that the data quality wasn't very high. For example for attendance and absence, I assumed that employees would enter hours religiously, however this is not the case. My results confirmed that the leavers had lower attendance and absence hours, which is probably due to lack of motivation and compliance. 

So I chose Random Forest Classifier above logistic regression and SVC. I tried the latter two algorithms (not included here) but RFC performed better. 

