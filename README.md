# From Coffee Machines to Machine Learning, Accenture
Fall 2023, AI Studio Project Write-Up

Table of Contents
I. Business Focus
II. Data Preparation and Validation
III. Approach
IV. Key Findings And Insights
V. Acknowledgements


# I. Business Focus
This project centers on applying machine learning methodologies to assist a client in the coffee shop industry in optimizing various facets of their business operations. The primary objectives include: 
1. Location Optimization:
Employing machine learning algorithms and pertinent datasets to pinpoint the most advantageous location for the client's coffee shop in the bustling landscape of New York City. Considerations revolve around variables such as population density, demographic insights, competitive analysis, foot traffic patterns, and proximity to key transportation hubs.
2. Specialty Items Selection:
Leveraging machine learning techniques to discern consumer preferences and current trends within the coffee shop market. Identifying three distinct specialty items that align with customer demands, enhancing the appeal and unique offerings of the coffee shop.
3. Business and Marketing Strategies:
Harnessing machine learning models and advanced algorithms to craft tailored marketing strategies for both the broader business and the specific coffee shop. Analyzing customer reviews to extract significant attributes correlated with successful and underperforming coffee shops, enabling data-driven decisions regarding operational and marketing priorities.
This project aims to harness the power of machine learning to optimize decision-making for the coffee shop, fostering customer satisfaction, refining business strategies, and driving growth in the competitive market.


# II. Data Preparation and Validation
DATASET DESCRIPTION
For the location model, our dataset encompasses diverse variables:
Business Information: ID, alias, name, review_count, categories, rating, transactions, location.zip_code, location.display_address.
Demographics Data: Extracted from 'demographics.xlsx', including Best Population Estimate.
The reviews model dataset comprises:
Business Details: Business_id, name, address, city, state, postal_code, latitude, longitude, stars_x, review_count, is_open, attributes, categories, hours, business_name.

DATASET PREPROCESSING
Location Model:
Initial Data Assessment:
Dropped irrelevant columns: 'is_closed', 'url', 'image_url', and others.
Merging Demographics Data:
Loaded and merged 'demographics.xlsx' based on the 'location.zip_code' and 'Geography' columns.
Replaced zero values with NaN for further processing.
Menu Model:
Yelp Dataset Integration:
Combined 'business.json' and 'reviews.json' from the Yelp Dataset.
Data Inspection:
Examined columns such as business_id, name, address, stars_y, useful, funny, cool, text, date, among others.
Filtering Relevant Data:
Focused on filtering reviews specifically relevant to coffee shops, employing a text corpus for extraction.
Marketing:
Unified Dataset:
Merged the location-based information with the filtered menu data to create a consolidated dataset for marketing analysis.

EDA (EXPLORATORY DATA ANALYSIS)
During the Data Understanding and Preparation phases, our team encountered substantial insights while exploring the datasets within the Google Colab environment. This exploratory phase underscored the pivotal role of comprehensive data exploration in shaping subsequent modeling and analysis. See our specific EDA in Google Colab, here: location_model and menu_model.
Location Model:
Insights from Census Data:
Demographic Analysis:
Leveraging EDA on the census dataset, we identified areas characterized by the highest income levels and racial demographics. Visualization techniques aided in understanding geographical nuances through insightful graphical representations.
Visualization Impact:
Visualization via correlation plots and graphs facilitated a deeper comprehension of our dataset, transcending the limitations of traditional CSV file scrutiny.
Challenges Encountered:
Yelp API Limitations:
Obtaining comprehensive data from the Yelp API posed challenges. The API's limitations, providing only 50 data points for cafes per call, led us to devise a workaround. Employing a for loop enabled multiple API calls, eventually creating our CSV file.
Review Data Limitation:
Restricted reviews per cafe (only 3 per API call) urged an alternative approach. We harnessed a Yelp dataset housing over 30,000 reviews, circumventing limitations inherent in the Yelp API.

Menu Model:
Attribute Extraction:
To understand the dataset's attributes, we extracted attribute names using the heart-disease.names file, employing regex to pinpoint the relevant sections. The attributes identified include:
Age: The individual's age in years.
Sex: Gender of the person (1 = male, 0 = female).
Chest Pain (cp): Description of the chest pain experienced.
Resting Blood Pressure (trestbps): Blood pressure measured on hospital admission (mm Hg).
Cholesterol (chol): Cholesterol measurement in mg/dl.
Fasting Blood Sugar (fbs): Presence of fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false).
Resting ECG (restecg): Electrocardiographic measurement.
Maximum Heart Rate Achieved (thalach): Individual's highest recorded heart rate.
Exercise-Induced Angina (exang): Presence of exercise-induced angina (1 = yes; 0 = no).
ST Depression Induced by Exercise (oldpeak): ST depression concerning rest during exercise.
Slope: Slope of the peak exercise ST segment.
Number of Major Vessels (ca): Count of major vessels (0-3).
Thalassemia (thal): Blood disorder categorization.
Target: Presence of heart disease (0-4, where 0 = no disease).
Data Import and Preprocessing:
File Import:
Data from processed.cleveland.data, processed.hungarian.data, processed.switzerland.data, and processed.va.data was imported for analysis.
Initial Data Assessment:
The Cleveland dataset was identified as the cleanest, serving as the base for further preprocessing and range determination for appropriate values.
Data Cleaning and Ranges Determination:
NaN and ? Values:
Rows containing NaN or '?' values were dropped.
Value Ranges:
Determined ranges for numerical attributes (age, trestbps, chol, thalach, oldpeak), ensuring data consistency and accuracy.
Data Type Conversion:
Conversion of appropriate columns to float64 dtype for uniform data representation.
Ongoing Cleaning and Refinement:
Cleanup Process:
Lists with values exceeding a threshold of 5 entries underwent cleanup, intending to maintain a manageable list size.
Further Data Cleaning:
Continued data cleaning processes, replacing specific values (e.g., '-9.0' treated as NaN) to enhance dataset reliability.
The EDA phase unveiled critical attributes and initiated data preprocessing steps essential for modeling and analysis, ensuring the dataset's quality and consistency.

 

FEATURE SELECTION
Location Model
To enhance our location model's predictive capability, we employed various techniques to select relevant features and generate new columns for evaluation:
Feature Engineering based on Zip Code:
We calculated an 'average rating' column for each coffee shop using the zip code information. This new feature helps determine the average rating based on coffee shops in the same area.
Aggregating Café Names by Zip Code:
We created an 'all_names' column listing all café names in a particular zip code area.
Optimal Location Labeling:
Introduced an 'optimal_location' binary classification based on predefined criteria like high average ratings and reviews to population ratio.
Data Refinement:
To enhance clarity and manageability, we rounded the 'Reviews to Population Ratio' to 3 decimal places and sorted the DataFrame for better analysis.

Menu Model:
For the menu model, we utilized feature selection techniques to identify the most significant attributes:
Selecting Best Features:
Employed SelectKBest from Scikit-learn to find the top k features using the chi-squared scoring function. This process aids in identifying the most influential features.
Feature Importance Analysis:
Utilized an ExtraTreesClassifier model to determine feature importances, generating a bar plot visualization for better comprehension.
Scatterplot Visualization:
Employed Plotly Express to create a scatterplot for the 'age' versus 'target' columns, visualizing their relationship.
These techniques facilitated the selection of relevant features crucial for both location and menu models, enhancing the predictive power and interpretability of our models.


# III. Approach
Location Model: Random Forest Binary Classifier
Objective: The location model aims to categorize coffee shop locations as 'optimal' or 'non-optimal' based on specific criteria like 'Reviews to Population Ratio' and 'average ratings' within zip codes.
Methodology:
Data Preparation:
Zip Code Grouping: Organized coffee shop data by zip codes.
Criterion Definition: Derived 'Reviews to Population Ratio' and 'average ratings' for each zip code.
Model Selection and Training:
Model Chosen: Employed a Random Forest Binary Classifier.
Feature Selection: Utilized 'review_count', 'Best Population Estimate', 'Reviews to Population Ratio', 'average rating'.
Training and Testing: Split the data into training and testing sets.
Hyperparameter Tuning:
Parameter Optimization: Explored hyperparameters for the Random Forest Classifier to enhance model performance.
Model Evaluation:
Performance Metrics: Assessed model accuracy, precision, recall, and F1-score.
Prediction Validation: Tested predictions for new location data.
Final Model Selection and Deployment:
Model Validation: Evaluated the best-performing model based on validation results.
Deployment Strategy: Prepared the model for deployment considering scalability and robustness.

Menu and Marketing Model: NLP Analysis of Reviews Data
Objective: The menu and marketing model focuses on using NLP techniques to derive insights from customer reviews, categorize them based on sentiment, and extract common words/phrases related to coffee shops.
Methodology:
Data Filtering and Preprocessing:
Keyword Filtering: Identified relevant keywords to filter coffee-related reviews.
Category Identification: Categorized reviews based on specific coffee-related categories.
NLP Analysis:
Keyword Extraction: Extracted food items and specific terms related to coffee from reviews.
Sentiment Analysis: Categorized reviews into positive, neutral, and negative sentiments.
Feature Extraction and Insights:
N-gram Analysis: Explored common word sequences (bigrams, trigrams) in different types of reviews.
Customer Sentiment Insights: Derived insights into customer preferences and sentiments based on review analysis.
Final Model Evaluation:
Review-Based Insights: Analyzed the most common words/phrases across different review types.
Final Insights for Marketing Strategy: Extracted actionable insights for menu and marketing strategy based on sentiment and customer preferences.


# IV. Key Findings And Insights
KEY RESULTS
Location Model Insights:
Our conclusive analysis derived from the location model identified three key areas—Soho, Williamsburg, and Midtown—as optimal coffee shop locations. These locations consistently demonstrated strong potential for successful coffee shop establishments, providing a robust foundation for strategic business placement. Soho, Williamsburg, and Midtown emerged as prime coffee shop locations following a comprehensive analysis. Soho's appeal lies in its high foot traffic, affluent demographics, and proximity to upscale stores, offering an ideal post-shopping relaxation spot. Williamsburg's vibrant arts scene and diverse community make it an attractive destination, while Midtown's central position and bustling commercial activity cater to a steady stream of commuters and professionals. However, while promising, these areas may face increased competition, requiring a thorough market evaluation to ensure sustained success in the competitive coffee shop landscape.

Menu Model Insights:
Furthermore, our investigation into preferred menu items underscored a significant correlation between specific choices and heightened customer satisfaction. Gooey butter cake, Cafe du Monde, and ice cream emerged as favored menu options, emphasizing a distinct inclination towards dessert choices over savory offerings. Complementing these findings, our research on key amenities highlighted crucial preferences among coffee shop-goers: an inclination towards intimate settings, efficient service with minimal wait times, and a diverse range of seating options. These insights collectively construct a comprehensive understanding of the elements that contribute to an optimal coffee shop experience, guiding strategic decisions in location selection and menu design.

Our menu model insights, generated through detailed NLP analysis involving bi, tri, and tetra-grams, provided a nuanced understanding of customer sentiments. Positive trends, including phrases like "get work done," "staff super friendly," and mentions of desirable seating options, were indicative of favorable customer experiences. Contrarily, neutral mentions of "drive-thru" and "almond milk," along with certain references to "Dunkin Donuts," conveyed less emotive or neutral sentiments. Conversely, negative sentiments encapsulated phrases highlighting issues such as "bad customer service" and "long wait times" that significantly impacted customer experiences. By amalgamating insights from our location and menu analyses, we meticulously crafted a blueprint defining the quintessential coffee shop—encompassing the top 20 foods correlated with high and low-star reviews, alongside positive and negative tetra-grams. These findings serve as a definitive guide for curating an appealing and customer-centric coffee shop experience, delineating the essential elements pivotal for success in this domain.

INSIGHTS
Significant Growth in Technical and Team Skills:
This project marked my initial venture into a substantial technical endeavor, and it has been pivotal in my growth. The collaborative environment exposed me to collective problem-solving, fostering innovative solutions that surpassed individual contributions. This collective approach not only elevated the project's quality but also cultivated a strong sense of camaraderie among team members.
Time Management Crucial for Success:
Balancing school commitments alongside this project demanded rigorous time management. Each team member consistently dedicated over three hours per week, often stretching into late hours to accommodate the project workload. Coordinating schedules for weekly meetings and finding additional work hours aligned with our varying schedules was a significant challenge that we collectively managed to navigate.
Importance of Seeking Help and Collaboration:
Acknowledging the significance of seeking guidance, our team actively engaged with the challenge advisor, teaching assistants, and utilized platforms like Slack for discussions and queries. These interactions provided invaluable insights, contributing significantly to our problem-solving process and overall project success. Regular team meetings, status updates, and openly sharing when we got stuck helped to ensure that everyone was on the same page. I am eager to bring these lessons learned to my internship this summer!.


# V. Acknowledgements
As our project culminates on this launch day, I'm astounded by the immense learning journey these past 10 weeks have offered. Heartfelt gratitude goes out to everyone at Vesta Corporation for their invaluable insights and guidance. I extend my sincerest appreciation to our Challenge Advisors, Timo and Celine, whose support has been instrumental. To my amazing teammates—Caroline, Farhin, Hafsa, Jing, and Steven—your collaborative spirit and dedication have been the bedrock of our achievements. Each one of you has enriched this experience profoundly.
I'd like to express my deepest thanks to our TA, Amber, for her unwavering assistance and course support. Additionally, immense gratitude goes to Break Through Tech, the Cornell Tech AI Program team, and specifically, Erika and Abby, for orchestrating an exceptional program experience. Your efforts have been invaluable in shaping this enriching journey. This experience has been transformative, and I'm excited to see the knowledge we've gained propel us forward. Here's to the future and the countless opportunities it holds!
