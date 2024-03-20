# Predictive-Maintenance
# machine learning algorithms
# Predictive Maintenance of Turbo fan Engine
# Linear Regression 
# Logistic Regression

# What is Turbofan Engine ?
The turbofan was invented to improve the fuel consumption of the turbojet. It achieves this by pushing more air, thus increasing the mass and lowering the speed of the propelling jet compared to that of the turbojet.
# Problem Statement of Predictive Maintenance
The task is to determine whether a Machine Learning model could be used to perform Predictive Maintenance on turbofan engines. For the purposes of this tutorial, we will assume that the following information has been ascertained through consultation with the company operating the turbofans:
1] The maintenance schedule of the turbofans is flexible. There would be no use carrying out this analysis if the schedule cannot be changed.
2] The analysis would generate long-term value for the operating company.
Given that the above points are true, the problem now lies in the analysis. We will use sensor data to predict the Remaining Useful Life (RUL) of turbofan engines. This RUL prediction can then be used to facilitate predictive maintenance.
The maintenance schedule of the turbofans is flexible. There would be no use carrying out this analysis if the schedule cannot be changed.
The analysis would generate long-term value for the operating company.
An explanation of what Predictive Maintenance is, and a demonstration of how a PdM algorithm may be implemented in the real world.
"Predictive maintenance (PdM) is maintenance that monitors the performance and condition of equipment during normal operation to reduce the likelihood of failures."
There are generally three different types of maintenance:

Reactive maintenance is the process of repairing assets to standard operating conditions after poor performance or breakdown is observed.
Preventive maintenance usually occurs on some type of schedule. Preventive maintenance is designed to keep machinery and parts in good condition but does not take the state or process into account.
Predictive maintenance occurs as needed, drawing on real-time collection and analysis of machine operation data to identify issues before they can interrupt production. With predictive maintenance, repairs happen during machine operation and address an actual problem. If a shutdown is required, it will be shorter and more targeted.
While the planned downtime in preventive maintenance may cause a decrease in overall capacity and/or availability, it is favoured over the unplanned downtime of reactive maintenance, where costs and duration may be unknown until the problem is diagnosed and addressed. It is also likely to interrupt other scheduling and planning which will cause further downstream time losses.
The aim of this post is to demystify some technical aspects of predictive maintenance through a Python solution to a real-world problem: turbofan engine degradation.

# Dataset Description
The data used in this notebook is based off a subset of the popular NASA Turbofan Engine Degradation Simulation Data Set. It contains data for 100 different turbofans.

"Engine degradation simulation was carried out using C-MAPSS. Four different sets were simulated (using set 3 here) under different combinations of operational conditions and fault modes. Records several sensor channels to characterize fault evolution. The data set was provided by the Prognostics CoE at NASA Ames."

The turbofan dataset features four datasets of increasing complexity (see table I) [2, 3]. The engines operate normally in the beginning but develop a fault over time. For the training sets, the engines are run to failure, while in the test sets the time series end ‘sometime’ before failure. The goal is to predict the Remaining Useful Life (RUL) of each turbofan engine.

Datasets include simulations of multiple turbofan engines over time, each row contains the following information:
1. Engine unit number
2. Time, in cycles
3. Three operational settings
4. 21 sensor readings

What I find really cool about this dataset is that you can’t use any domain knowledge, as you don’t know what a sensor has been measuring. So, results are purely based on applying the correct techniques.

In today’s post we’ll focus on exploring the first dataset (FD001) in which all engines develop the same fault and have only one operating condition. In addition, we’ll create a baseline linear regression model so we can compare our modeling efforts of future posts.

Datasets include simulations of multiple turbofan engines over time, each row contains the following information:
1. Engine unit number
2. Time, in cycles
3. Three operational settings
4. 21 sensor readings

What I find really cool about this dataset is that you can’t use any domain knowledge, as you don’t know what a sensor has been measuring. So, results are purely based on applying the correct techniques.

In today’s post we’ll focus on exploring the first dataset (FD001) in which all engines develop the same fault and have only one operating condition. In addition, we’ll create a baseline linear regression model so we can compare our modeling efforts of future posts.


Given that the above points are true, the problem now lies in the analysis. We will use sensor data to predict the Remaining Useful Life (RUL) of turbofan engines. This RUL prediction can then be used to facilitate predictive maintenance.
Predictive maintenance of turbofan engines is the practice of determining the condition of equipment in order to estimate when maintenance should be performed. This concept aims to minimize the engine down-time and maintenance costs while preserving its required performance level and airworthiness. Artificial intelligence (AI) and Machine Learning in predictive maintenance, combined with condition status monitoring and health status monitoring, have changed the engine maintenance landscape. By tracking critical engine performance parameters for ensuring engine health like thrust, fuel flow, exhaust gas temperature (EGT), and compressor discharge pressure, unplanned and costly shutdowns transform into strategic, well-timed maintenance activities. The cornerstone of this transformation is the ability to accurately predict a range of turbofan engine failures using advanced Machine Learning algorithms.

This ability enables stakeholders to make data-driven decisions about the timing and nature of maintenance interventions. Since aircraft engines play a critical role in both safety and operational performance the need for reliable predictive models, powered by AI, is pressing. Moreover, predictive maintenance goes beyond just ensuring engine operational efficiency; it is pivotal in minimizing emissions and ensuring that engines operate within environmentally sustainable parameters.The task of accurately forecasting turbofan engine failures hinges on inherent challenges related to data quality and associated operational risks:
Obtaining real-time data for actual failures is fraught with risk, including the risk of an engine crash. As a result, most of the data used to train models is generated in the lab under controlled conditions and is less reflective of real-world scenarios.
# The Solution of Problem 
Our solution is a customized software model designed to mitigate the challenges outlined above. Features of our model include:

• Advanced sequence analysis: Our model includes the ability to accept sequences, which are time series representations of sensor signal vectors. This allows the model to detect and identify trends and patterns within the signals to improve the quality of the data used to make predictions.

• Data filtering: The model filters out irrelevant data, focusing only on significant values for more accurate predictions.

• Data Points: We accrue a variety of engine condition parameters collected during the flight, including pressure, vibration, and oil temperature.

• Remaining life prediction: The model foretells the remaining useful life (RUL) of turbofan engines, integrating condition status monitoring insights.

• Prediction for multiple engines: Our system can make predictions of current quality for an entire fleet of engines based on the latest sensor data, flagging engines that are likely to fail soon due to discrepancies in critical engine performance parameters.

• Trend prediction: The model also provides a trend prediction feature that uses a range of sensor data to increase confidence in the future performance of specific engines.

• Transparency: We provide transparent access to the raw prediction data, enabling further analysis and refinement by the end user.
# Overview
Although released over a decade ago, NASA’s turbofan engine degradation simulation dataset (CMAPSS) remains popular and relevant today. Over 90 new research papers have been published in 2020 so far [1]. These papers present and benchmark novel algorithms to predict Remaining Useful Life (RUL) on the turbofan datasets.

When I first started learning about predictive maintenance, I stumbled upon a few blog posts using the turbofan degradation dataset. Each covered exploratory data analysis and a simple model to predict the RUL, but I felt two things were lacking:

I only focused on the first dataset, leaving me guessing how the more complex challenges could be solved.
A few years later this seemed like a fun project for me to pick up. In a series of posts, I plan to showcase and explain multiple analysis techniques, while also offering a solution for the more complex datasets.
