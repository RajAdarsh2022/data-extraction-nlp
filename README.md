# Blackoffer Consulting NLP Project

This repository contains code and analysis related to performing Textual Sentiment Analysis on articles published on the Blackoffer Consulting website.

## Overview

The goal of this project is to extract article text from the Blackoffer Consulting website, perform sentiment analysis, and generate various scores including Positive Score, Negative Score, Polarity Score, Subjectivity Score, etc. The analysis is carried out using Python's NLTK library.

## Data Extraction

Article text and title were extracted using BeautifulSoup from the URLs listed in the `input.xlsx` file. The extraction process excluded website headers, footers, and any content other than the article text.

## Data Preprocessing

Data preprocessing steps were carried out, including the removal of stopwords. Sentiment analysis was performed using Python and NLTK. The final results are saved in the `Output_Data_Structure.xlsx` file.

## Screenshots

![Table](https://raw.githubusercontent.com/RajAdarsh2022/data-extraction-nlp/main/assets/screenshots/data-extraction-nlp_ss.png)



 
