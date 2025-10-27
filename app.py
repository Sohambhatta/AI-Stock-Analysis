"""
Stock Market Prediction Web Application

A Flask-based web application that provides stock market analysis and predictions
by combining technical analysis with news sentiment analysis. The application
fetches real-time stock data, analyzes recent news articles, and generates
investment recommendations.

Author: Soham Bhatta
Version: 2.0
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import yfinance as yf
from newsapi import NewsApiClient
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from datetime import datetime
import feedparser

# Initialize Flask application and configure CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

# Initialize sentiment analysis tools
sentiment_analyzer = SentimentIntensityAnalyzer()

# News API configuration - supports both NewsAPI and RSS fallback
# To use NewsAPI, set the NEWS_API_KEY environment variable
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_newsapi_key_here')
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY != 'your_newsapi_key_here' else None

# Global error handlers to ensure consistent JSON responses
@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 errors by returning JSON response instead of HTML"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def handle_internal_error(error):
    """Handle 500 errors by returning JSON response instead of HTML"""
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(Exception)
def handle_general_exception(error):
    """Handle any unhandled exceptions by returning JSON response"""
    return jsonify({'error': str(error)}), 500

# Popular stocks for quick access on the dashboard
# These are commonly traded stocks that users frequently analyze
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms Inc.',
    'NFLX': 'Netflix Inc.',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.'
}

# Extended list of stocks for search suggestions
# Includes major stocks from various sectors for comprehensive search functionality
STOCK_SUGGESTIONS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'GOOG': 'Alphabet Inc. Class C',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms Inc.',
    'NFLX': 'Netflix Inc.',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble',
    'UNH': 'UnitedHealth Group',
    'HD': 'Home Depot',
    'MA': 'Mastercard Inc.',
    'BAC': 'Bank of America',
    'DIS': 'Walt Disney Co.',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'PYPL': 'PayPal Holdings',
    'INTC': 'Intel Corp.',
    'CMCSA': 'Comcast Corp.',
    'PEP': 'PepsiCo Inc.',
    'T': 'AT&T Inc.',
    'ABT': 'Abbott Laboratories',
    'COP': 'ConocoPhillips',
    'TMO': 'Thermo Fisher Scientific',
    'COST': 'Costco Wholesale',
    'LLY': 'Eli Lilly and Co.',
    'ACN': 'Accenture PLC',
    'AVGO': 'Broadcom Inc.',
    'DHR': 'Danaher Corp.',
    'VZ': 'Verizon Communications',
    'TXN': 'Texas Instruments',
    'NEE': 'NextEra Energy',
    'HON': 'Honeywell International',
    'QCOM': 'Qualcomm Inc.',
    'PM': 'Philip Morris International',
    'SPGI': 'S&P Global Inc.',
    'LOW': 'Lowe\'s Companies',
    'MDT': 'Medtronic PLC',
    'UNP': 'Union Pacific Corp.',
    'BMY': 'Bristol-Myers Squibb',
    'IBM': 'International Business Machines',
    'RTX': 'Raytheon Technologies',
    'AMD': 'Advanced Micro Devices',
    'ORCL': 'Oracle Corp.',
    'BLK': 'BlackRock Inc.',
    'CVX': 'Chevron Corp.',
    'CAT': 'Caterpillar Inc.',
    'GS': 'Goldman Sachs Group',
    'SBUX': 'Starbucks Corp.',
    'BA': 'Boeing Co.',
    'AMT': 'American Tower Corp.',
    'LMT': 'Lockheed Martin Corp.',
    'AXP': 'American Express Co.',
    'MU': 'Micron Technology',
    'DE': 'Deere & Co.',
    'BKNG': 'Booking Holdings Inc.',
    'GILD': 'Gilead Sciences',
    'MDLZ': 'Mondelez International',
    'ADP': 'Automatic Data Processing',
    'TJX': 'TJX Companies',
    'SYK': 'Stryker Corp.',
    'ZTS': 'Zoetis Inc.',
    'MMM': '3M Co.',
    'CVS': 'CVS Health Corp.',
    'TMUS': 'T-Mobile US Inc.',
    'MO': 'Altria Group Inc.',
    'PFE': 'Pfizer Inc.',
    'C': 'Citigroup Inc.',
    'AON': 'Aon PLC',
    'NOW': 'ServiceNow Inc.',
    'ISRG': 'Intuitive Surgical',
    'CB': 'Chubb Ltd.',
    'MMC': 'Marsh & McLennan',
    'USB': 'U.S. Bancorp',
    'CSX': 'CSX Corp.',
    'CL': 'Colgate-Palmolive Co.',
    'FDX': 'FedEx Corp.',
    'WM': 'Waste Management',
    'ICE': 'Intercontinental Exchange',
    'NSC': 'Norfolk Southern Corp.',
    'PLD': 'Prologis Inc.',
    'APD': 'Air Products and Chemicals',
    'COF': 'Capital One Financial',
    'FISV': 'Fiserv Inc.',
    'TGT': 'Target Corp.',
    'KMI': 'Kinder Morgan Inc.',
    'GE': 'General Electric Co.',
    'SHW': 'Sherwin-Williams Co.',
    'CME': 'CME Group Inc.',
    'DUK': 'Duke Energy Corp.',
    'SO': 'Southern Co.',
    'ITW': 'Illinois Tool Works',
    'TFC': 'Truist Financial Corp.',
    'PNC': 'PNC Financial Services',
    'EMR': 'Emerson Electric Co.',
    'BSX': 'Boston Scientific Corp.'
}

def get_stock_news(symbol, company_name):
    """
    Fetch recent news articles related to a specific stock symbol.
    
    This function attempts to gather news from multiple sources:
    1. Yahoo Finance RSS feeds for stock-specific news
    2. Google News RSS feeds for broader company coverage
    3. Fallback to synthetic news based on price movements if no articles found
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        company_name (str): Full company name for search queries
    
    Returns:
        list: List of dictionaries containing news articles with keys:
              - title: Article headline
              - description: Article summary/description
              - url: Link to full article
              - published_at: Publication date
              - source: News source name
    """
    try:
        articles = []
        
        # Primary news source: Yahoo Finance RSS for stock-specific news
        # This provides the most relevant financial news for the specific ticker
        try:
            yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            yahoo_feed = feedparser.parse(yahoo_url)
            
            # Process Yahoo Finance articles (typically most relevant)
            for entry in yahoo_feed.entries[:3]:  # Limit to top 3 articles
                if hasattr(entry, 'title') and hasattr(entry, 'summary'):
                    articles.append({
                        'title': entry.title,
                        'description': entry.summary[:300],  # Truncate for display
                        'url': entry.link if hasattr(entry, 'link') else '',
                        'published_at': entry.published if hasattr(entry, 'published') else '',
                        'source': 'Yahoo Finance'
                    })
        except Exception as e:
            print(f"Yahoo Finance RSS error for {symbol}: {e}")
        
        # Secondary news source: Google News for broader company coverage
        # This helps capture news that might not be in Yahoo Finance feeds
        try:
            search_terms = [symbol, company_name.split()[0]]  # Search by symbol and company name
            
            for term in search_terms:
                if len(articles) >= 5:  # Limit total articles to prevent information overload
                    break
                    
                # Query Google News RSS for stock-related articles
                google_news_url = f"https://news.google.com/rss/search?q={term}+stock&hl=en-US&gl=US&ceid=US:en"
                
                try:
                    google_feed = feedparser.parse(google_news_url)
                    
                    # Filter articles for relevance to our specific stock
                    for entry in google_feed.entries[:2]:  # Limit per search term
                        if len(articles) >= 5:
                            break
                            
                        if hasattr(entry, 'title') and hasattr(entry, 'summary'):
                            # Check if article is actually relevant to our stock
                            title_lower = entry.title.lower()
                            if symbol.lower() in title_lower or company_name.split()[0].lower() in title_lower:
                                articles.append({
                                    'title': entry.title,
                                    'description': entry.summary[:300] if hasattr(entry, 'summary') else entry.title,
                                    'url': entry.link if hasattr(entry, 'link') else '',
                                    'published_at': entry.published if hasattr(entry, 'published') else '',
                                    'source': 'Google News'
                                })
                except Exception as e:
                    print(f"Google News RSS error for {term}: {e}")
                    
        except Exception as e:
            print(f"Google News search error for {symbol}: {e}")
        
        # Fallback method: Generate synthetic news based on recent price action
        # This ensures we always have some form of news context for analysis
        if len(articles) == 0:
            try:
                # Analyze recent stock performance to create basic market commentary
                stock = yf.Ticker(symbol)
                hist = stock.history(period="5d")
                
                if not hist.empty:
                    # Calculate recent price performance
                    recent_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    
                    # Only generate synthetic news for significant price movements
                    if abs(recent_change) > 1:
                        direction = "surged" if recent_change > 0 else "declined"
                        sentiment_word = "positive" if recent_change > 0 else "negative"
                        
                        articles.append({
                            'title': f"{company_name} stock {direction} {abs(recent_change):.1f}% in recent trading",
                            'description': f"Market data shows {symbol} has {direction} {abs(recent_change):.1f}% over the past few trading sessions, indicating {sentiment_word} market sentiment among investors.",
                            'url': f"https://finance.yahoo.com/quote/{symbol}",
                            'published_at': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'Market Data Analysis'
                        })
            except Exception as e:
                print(f"Fallback news generation error for {symbol}: {e}")
        
        return articles[:5]  # Return maximum of 5 articles for optimal performance
        
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using multiple sentiment analysis methods.
    
    This function combines two different sentiment analysis approaches:
    1. VADER (Valence Aware Dictionary and sEntiment Reasoner) - optimized for social media text
    2. TextBlob - general purpose sentiment analysis with polarity scoring
    
    The combination provides more robust sentiment analysis than either method alone.
    
    Args:
        text (str): Text to analyze for sentiment
    
    Returns:
        dict: Dictionary containing:
            - score: Combined sentiment score (-1 to 1)
            - label: Human-readable sentiment ('positive', 'negative', 'neutral')
            - vader: Raw VADER compound score
            - textblob: Raw TextBlob polarity score
    """
    if not text:
        return {'score': 0, 'label': 'neutral'}
    
    # VADER sentiment analysis - better for informal text and social media
    # Returns compound score between -1 (most negative) and 1 (most positive)
    vader_scores = sentiment_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    
    # TextBlob sentiment analysis - general purpose polarity detection
    # Returns polarity between -1 (negative) and 1 (positive)
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    
    # Combine scores using weighted average (VADER weighted higher for financial text)
    # VADER is given more weight as it handles financial/news text better
    combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)
    
    # Determine sentiment label based on combined score
    # Using thresholds to avoid neutral classification for weak signals
    if combined_score >= 0.1:
        label = 'positive'
    elif combined_score <= -0.1:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'score': round(combined_score, 3),
        'label': label,
        'vader': round(vader_compound, 3),
        'textblob': round(textblob_polarity, 3)
    }

def get_news_sentiment(symbol, company_name):
    """
    Analyze sentiment of news articles for a specific stock.
    
    This function fetches news articles and performs comprehensive sentiment analysis
    to determine overall market sentiment towards the stock. It processes each article
    individually and then calculates aggregate sentiment metrics.
    
    Args:
        symbol (str): Stock ticker symbol
        company_name (str): Company name for news search
    
    Returns:
        dict: Dictionary containing:
            - articles: List of analyzed articles with individual sentiment scores
            - overall_sentiment: Aggregate sentiment analysis across all articles
            - sentiment_summary: Text summary of sentiment analysis results
    """
    # Fetch recent news articles for the stock
    articles = get_stock_news(symbol, company_name)
    
    # Handle case where no news articles are available
    if not articles:
        return {
            'articles': [],
            'overall_sentiment': {'score': 0, 'label': 'neutral'},
            'sentiment_summary': 'No recent news available for analysis.'
        }
    
    analyzed_articles = []
    sentiment_scores = []
    
    # Process each article for sentiment analysis
    for article in articles:
        # Combine title and description for comprehensive sentiment analysis
        # Title often contains the key sentiment, description provides context
        text = f"{article['title']} {article['description']}"
        sentiment = analyze_sentiment(text)
        
        # Store article with sentiment analysis results
        analyzed_articles.append({
            'title': article['title'],
            'description': article['description'][:200] + '...' if len(article['description']) > 200 else article['description'],
            'sentiment': sentiment,
            'source': article['source'],
            'published_at': article['published_at']
        })
        
        # Collect sentiment scores for overall calculation
        sentiment_scores.append(sentiment['score'])
    
    # Calculate overall sentiment from individual article scores
    if sentiment_scores:
        # Use average of all sentiment scores
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        overall_sentiment = {'score': round(avg_sentiment, 3)}
        
        # Classify overall sentiment based on average score
        if avg_sentiment >= 0.1:
            overall_sentiment['label'] = 'positive'
        elif avg_sentiment <= -0.1:
            overall_sentiment['label'] = 'negative'
        else:
            overall_sentiment['label'] = 'neutral'
    else:
        overall_sentiment = {'score': 0, 'label': 'neutral'}
    
    # Generate summary statistics for sentiment distribution
    positive_count = sum(1 for score in sentiment_scores if score > 0.1)
    negative_count = sum(1 for score in sentiment_scores if score < -0.1)
    neutral_count = len(sentiment_scores) - positive_count - negative_count
    
    sentiment_summary = f"Analysis of {len(articles)} recent articles: {positive_count} positive, {negative_count} negative, {neutral_count} neutral."
    
    return {
        'articles': analyzed_articles,
        'overall_sentiment': overall_sentiment,
        'sentiment_summary': sentiment_summary
    }

@app.route('/')
def index():
    """
    Serve the main application page.
    
    Returns the main HTML template with the list of popular stocks
    for quick access on the dashboard.
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html', popular_stocks=POPULAR_STOCKS)

@app.route('/api/stock_suggestions/<query>')
def get_stock_suggestions(query):
    """
    Provide stock symbol suggestions based on user input.
    
    This endpoint searches through the stock database to find symbols
    and company names that match the user's query. Used for autocomplete
    functionality in the search interface.
    
    Args:
        query (str): User's search query
    
    Returns:
        JSON: List of matching stocks with symbol and name
    """
    query = query.upper()
    suggestions = []
    
    # Search through available stocks for matches
    for symbol, name in STOCK_SUGGESTIONS.items():
        # Match against both symbol and company name
        if symbol.startswith(query) or query in name.upper():
            suggestions.append({
                'symbol': symbol,
                'name': name
            })
            # Limit results to prevent overwhelming the interface
            if len(suggestions) >= 8:
                break
    
    return jsonify(suggestions)

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """
    Comprehensive stock analysis endpoint that combines technical and sentiment analysis.
    
    This is the main analysis function that fetches stock data, performs technical analysis,
    gathers news sentiment, and generates investment recommendations. The analysis combines
    multiple data sources to provide a well-rounded view of the stock's prospects.
    
    Args:
        symbol (str): Stock ticker symbol to analyze
    
    Returns:
        JSON: Complete stock analysis including:
            - Basic stock information (price, change, etc.)
            - Chart data for visualization
            - Technical analysis scores
            - News sentiment analysis
            - Combined prediction and confidence score
            - Company fundamentals (market cap, P/E ratio, etc.)
    """
    try:
        symbol = symbol.upper()
        print(f"Beginning analysis for {symbol}...")
        
        # Fetch historical stock data using yfinance
        # 6 months provides good context for technical analysis
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        # Validate that we have data for this symbol
        if hist.empty:
            return jsonify({
                'error': f'No data found for symbol {symbol}. Please verify the symbol is correct.'
            }), 400
        
        # Attempt to fetch company information and fundamentals
        # Some stocks may not have complete fundamental data
        try:
            info = stock.info
            company_name = info.get('longName', info.get('shortName', symbol))
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            dividend_yield = info.get('dividendYield', 'N/A')
        except Exception as e:
            print(f"Could not fetch company fundamentals for {symbol}: {e}")
            # Use defaults when fundamental data is unavailable
            company_name = symbol
            market_cap = 'N/A'
            pe_ratio = 'N/A'
            dividend_yield = 'N/A'
        
        # Prepare chart data for frontend visualization
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        
        # Calculate current price and recent performance metrics
        current_price = prices[-1] if prices else 0
        price_change = prices[-1] - prices[-2] if len(prices) > 1 else 0
        price_change_pct = (price_change / prices[-2] * 100) if len(prices) > 1 and prices[-2] != 0 else 0
        
        # Get news sentiment analysis
        print(f"Analyzing news sentiment for {symbol}...")
        news_data = get_news_sentiment(symbol, company_name)
        
        # Technical analysis
        recent_prices = prices[-5:]  # Last 5 days
        technical_score = 0
        technical_sentiment = "NEUTRAL"
        
        if len(recent_prices) >= 5:
            trend = sum(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices)))
            
            if trend >= 4:  # 4 out of 5 days up
                technical_score = 0.7
                technical_sentiment = "POSITIVE"
            elif trend <= 1:  # 1 or 0 days up
                technical_score = -0.7
                technical_sentiment = "NEGATIVE"
            else:
                technical_score = 0
                technical_sentiment = "NEUTRAL"
        else:
            # Fallback to simple price change
            if price_change_pct > 2:
                technical_score = 0.6
                technical_sentiment = "POSITIVE"
            elif price_change_pct < -2:
                technical_score = -0.6
                technical_sentiment = "NEGATIVE"
            else:
                technical_score = 0
                technical_sentiment = "NEUTRAL"
        
        # Combine technical analysis with news sentiment
        news_sentiment_score = news_data['overall_sentiment']['score']
        
        # Weighted combination: 60% technical, 40% news sentiment
        combined_score = (technical_score * 0.6) + (news_sentiment_score * 0.4)
        
        # Determine final prediction
        if combined_score > 0.3:
            prediction = "BUY"
            sentiment = "POSITIVE"
            base_confidence = 75
        elif combined_score < -0.3:
            prediction = "SELL"
            sentiment = "NEGATIVE"
            base_confidence = 75
        elif combined_score > 0.1:
            prediction = "BUY"
            sentiment = "POSITIVE"
            base_confidence = 60
        elif combined_score < -0.1:
            prediction = "SELL"
            sentiment = "NEGATIVE"
            base_confidence = 60
        else:
            prediction = "HOLD"
            sentiment = "NEUTRAL"
            base_confidence = 50
        
        # Adjust confidence based on news availability and price volatility
        confidence = base_confidence
        if len(news_data['articles']) > 0:
            confidence += 10  # Boost confidence when we have news data
        
        confidence += min(15, abs(price_change_pct) * 2)  # Boost based on price movement
        confidence = min(95, max(30, confidence))  # Cap between 30-95%
        
        # Create enhanced reasoning
        reasoning = f"Combined analysis: Technical indicators show {technical_sentiment.lower()} signals "
        reasoning += f"({technical_score:+.2f}), while news sentiment is {news_data['overall_sentiment']['label']} "
        reasoning += f"({news_sentiment_score:+.2f}). "
        
        if abs(combined_score) > 0.5:
            reasoning += f"Strong {'bullish' if combined_score > 0 else 'bearish'} consensus detected. "
        elif abs(combined_score) > 0.2:
            reasoning += f"Moderate {'positive' if combined_score > 0 else 'negative'} signals. "
        else:
            reasoning += "Mixed signals suggest cautious approach. "
        
        reasoning += news_data['sentiment_summary']
        
        # Prepare news summary with actual quotes
        news_summary = []
        if news_data['articles']:
            news_summary.append(f"Recent news sentiment: {news_data['overall_sentiment']['label']} ({news_sentiment_score:+.2f})")
            
            # Add top 3 news headlines with sentiment
            for i, article in enumerate(news_data['articles'][:3]):
                sentiment_label = article['sentiment']['label']
                sentiment_emoji = "📈" if sentiment_label == 'positive' else "📉" if sentiment_label == 'negative' else "➡️"
                news_summary.append(f"{sentiment_emoji} \"{article['title']}\" - {article['source']} ({sentiment_label})")
        else:
            news_summary.append("No recent news available for sentiment analysis")
            news_summary.append(f"Technical analysis shows {technical_sentiment.lower()} trend")
            news_summary.append(f"Stock has {'gained' if price_change > 0 else 'lost'} {abs(price_change_pct):.2f}% recently")
        
        # Prepare response
        response_data = {
            'symbol': symbol,
            'name': company_name,
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'chart_data': {
                'dates': dates,
                'prices': prices,
                'volumes': volumes
            },
            'prediction': {
                'action': prediction,
                'confidence': int(confidence),
                'sentiment': sentiment,
                'reasoning': reasoning,
                'news_summary': news_summary,
                'technical_score': round(technical_score, 3),
                'news_sentiment_score': round(news_sentiment_score, 3),
                'combined_score': round(combined_score, 3)
            },
            'news_analysis': {
                'overall_sentiment': news_data['overall_sentiment'],
                'articles_analyzed': len(news_data['articles']),
                'sentiment_summary': news_data['sentiment_summary'],
                'articles': news_data['articles'][:3]  # Include top 3 articles
            },
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield
        }
        
        print(f"Successfully fetched data for {symbol}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f'Error fetching {symbol}: {str(e)}'
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/popular_stocks')
def get_popular_stocks():
    """Get quick data for popular stocks"""
    stocks_data = []
    for symbol, name in POPULAR_STOCKS.items():
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")
            
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price * 100) if prev_price != 0 else 0
                
                stocks_data.append({
                    'symbol': symbol,
                    'name': name,
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2)
                })
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            # Add placeholder data
            stocks_data.append({
                'symbol': symbol,
                'name': name,
                'price': 0.00,
                'change': 0.00,
                'change_pct': 0.00
            })
            
    return jsonify(stocks_data)

if __name__ == '__main__':
    print("Starting Stock Market Prediction Application...")
    print("Server running at: http://localhost:5000")
    print("Application Features:")
    print("  - Real-time stock data and charts")
    print("  - Technical analysis with news sentiment") 
    print("  - Popular stocks dashboard")
    print("  - Stock search with autocomplete")
    print("  - Investment recommendations with confidence scoring")
    app.run(debug=True, host='0.0.0.0', port=5000)