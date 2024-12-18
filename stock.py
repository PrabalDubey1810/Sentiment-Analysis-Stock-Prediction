import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER Lexicon
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Example Financial News Headlines
data = {
    'Date': ['2024-12-10', '2024-12-11', '2024-12-12', '2024-12-13', '2024-12-14'],
    'Headline': [
        "Company X reports record earnings in Q4!",
        "Market downturn expected due to economic uncertainty",
        "Company Y announces groundbreaking AI technology",
        "Stock prices plummet amid regulatory concerns",
        "Positive outlook for tech stocks as demand surges"
    ]
}

# Convert to DataFrame
news_df = pd.DataFrame(data)

# Ensure 'Date' column is in datetime format for proper handling
news_df['Date'] = pd.to_datetime(news_df['Date'])

# Add Sentiment Scores
news_df['Sentiment_Score'] = news_df['Headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Load Stock Data (Example: SPY - S&P 500 ETF)
start_date = "2024-12-09"
end_date = "2024-12-15"
stock_data = yf.download("SPY", start=start_date, end=end_date)

# Ensure 'Date' column is in datetime format for proper handling
stock_data['Date'] = stock_data.index
stock_data.reset_index(drop=True, inplace=True)

# Calculate Stock Price Movement (Close - Open) without merging
stock_data['Price_Change'] = stock_data['Close'] - stock_data['Open']
stock_data['Trend'] = stock_data['Price_Change'].apply(lambda x: 'Up' if x > 0 else 'Down')

# Plot Sentiment vs Stock Price Movement (Separate Plots)
plt.figure(figsize=(14, 6))

# Subplot for Sentiment Score
plt.subplot(1, 2, 1)
plt.scatter(news_df['Date'], news_df['Sentiment_Score'], c=news_df['Sentiment_Score'], cmap='coolwarm', edgecolor='k')
plt.title('Sentiment Score Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.colorbar(label='Sentiment Score')
plt.grid(True, alpha=0.5)

# Subplot for Stock Price Change
plt.subplot(1, 2, 2)
plt.scatter(stock_data['Date'], stock_data['Price_Change'], c=stock_data['Price_Change'], cmap='coolwarm', edgecolor='k')
plt.title('Stock Price Change Over Time', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Stock Price Change')
plt.colorbar(label='Price Change')
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# Display Data
print("News DataFrame with Sentiment Scores:")
print(news_df[['Date', 'Headline', 'Sentiment_Score']])

print("\nStock DataFrame with Price Changes:")
print(stock_data[['Date', 'Open', 'Close', 'Price_Change', 'Trend']])
