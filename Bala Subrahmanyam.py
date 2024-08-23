#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install vaderSentiment
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.parse
def generate_image_description(image_url):
    api_key = 'openai/clip-vit-base-patch16'
    url = 'https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    payload = {
        'inputs': image_url
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        if response.status_code == 200:
            return response.json()['generated_text']
        else:
            print(f"Failed to generate description. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during request: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

# Function to analyze emotional tone using sentiment analysis
def analyze_emotional_tone(description):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(description)
    
    # Determine emotional tone based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    image_url = input("Enter the URL of the image: ").strip()
    try:
        urllib.parse.urlparse(image_url)
    except Exception as e:
        print(f"Invalid URL: {str(e)}")
        return
    description = generate_image_description(image_url)
    if description:
        print("\nGenerated Description:", description)
        emotional_tone = analyze_emotional_tone(description)
        print("Emotional Tone:", emotional_tone)
    else:
        print("Failed to generate description.")

if __name__ == "__main__":
    main()




# In[ ]:


pip install vaderSentiment


# In[ ]:





# In[ ]:




