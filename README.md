# Stock Market Prediction App

I built this web application to help analyze stocks by combining technical analysis with news sentiment. The idea was to create something that looks at price charts and also considers what's happening in the news around a stock.

## What I Built

I wanted to create a tool that gives me more than just basic stock prices. Here's what I included:

- Real-time stock data that I pull from Yahoo Finance
- Interactive charts so I can see 6 months of price history
- Technical analysis that looks at recent price trends
- News analysis that actually reads recent articles and figures out if they're positive or negative
- A prediction system that combines both technical and news data
- A dashboard with popular stocks for quick access
- Search functionality with autocomplete

The nice part is that I made it work without needing any paid APIs - everything uses free sources.

## How to Run It

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python app.py
   ```

3. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## The Tech Stack I Used

I picked these because they're reliable and I've worked with them before:

**Backend:**
- Flask for the web framework
- yfinance to get stock data from Yahoo Finance
- feedparser to read RSS feeds for news
- textblob and vaderSentiment for analyzing news sentiment

**Frontend:**
- Just HTML/CSS/JavaScript (kept it simple)
- Plotly.js for the interactive charts

## How I Made the Analysis Work

I spent time figuring out how to make predictions that actually make sense:

### Stock Data
I pull 6 months of historical data for each stock, plus current market info like market cap and P/E ratios when available.

### News Collection
This was tricky but I found a way to get news without paying for APIs:
- I scrape Yahoo Finance RSS feeds for stock-specific news
- I also check Google News for broader company coverage
- If there's no news available, I create basic market commentary based on recent price changes

### Sentiment Analysis
I use two different methods and combine them:
- VADER (works well with financial/social media text)
- TextBlob (general purpose sentiment analysis)
- I weight VADER higher since it handles financial language better

### Making Predictions
My algorithm combines:
- Technical analysis (60% weight) - looks at recent price trends
- News sentiment (40% weight) - average sentiment from recent articles
- I calculate a confidence score based on how much the technical and news analysis agree

## What You'll See

When you analyze a stock, I show you:
- Current price and recent changes
- An interactive price chart
- My buy/sell/hold recommendation with confidence percentage
- A breakdown showing the technical score vs news sentiment score
- Actual news headlines with their individual sentiment scores
- Company fundamentals when available

The confidence levels I use:
- 70-95%: I'm pretty confident in this prediction
- 50-70%: Mixed signals, be cautious
- 30-50%: Not enough clear data to be confident

## Important Note

I built this as a learning project to understand how stock analysis works. It's not meant to be used as the only way to make investment decisions. Always do your own research and consider multiple factors before investing.
