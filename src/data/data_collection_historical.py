import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
from newspaper import Article
import arxiv
from bs4 import BeautifulSoup
import logging

class HistoricalNewsCollector:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.logger = logging.getLogger(__name__)

    def fetch_marketaux_historical(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical news using Marketaux API (formerly FMP)
        """
        base_url = "https://api.marketaux.com/v1/news/all"
        news_data = []
        
        try:
            params = {
                "api_token": self.api_keys["marketaux"],
                "symbols": symbol,
                "filter_entities": True,
                "from": start_date,
                "to": end_date,
                "limit": 100
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if "data" in data:
                for article in data["data"]:
                    news_data.append({
                        "date": article["published_at"],
                        "title": article["title"],
                        "source": article["source"],
                        "url": article["url"],
                        "sentiment": article["entities"][0]["sentiment_score"] if article["entities"] else None
                    })
            
            return pd.DataFrame(news_data)
        
        except Exception as e:
            self.logger.error(f"Error fetching Marketaux data: {str(e)}")
            return pd.DataFrame()

    def fetch_financial_times_archive(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical news from Financial Times Archive API
        """
        base_url = "https://api.ft.com/content/search/v1"
        news_data = []
        
        try:
            headers = {
                "X-Api-Key": self.api_keys["ft_archive"]
            }
            
            body = {
                "queryString": f""{symbol}"",
                "queryContext": {
                    "curations": ["ARTICLES"]
                },
                "resultContext": {
                    "aspects": ["title", "summary", "location"],
                    "sortField": "lastPublishDateTime",
                    "sortOrder": "DESC",
                    "dateRange": {
                        "from": start_date,
                        "to": end_date
                    }
                }
            }
            
            response = requests.post(base_url, headers=headers, json=body)
            data = response.json()
            
            if "results" in data:
                for article in data["results"]:
                    news_data.append({
                        "date": article["lastPublishDateTime"],
                        "title": article["title"]["title"],
                        "source": "Financial Times",
                        "url": article["location"]["uri"]
                    })
            
            return pd.DataFrame(news_data)
            
        except Exception as e:
            self.logger.error(f"Error fetching FT archive data: {str(e)}")
            return pd.DataFrame()

    def fetch_sec_filings(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical SEC filings using EDGAR
        """
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        news_data = []
        
        try:
            params = {
                "CIK": symbol,
                "type": "8-K",
                "dateb": end_date.replace("-", ""),
                "datea": start_date.replace("-", ""),
                "owner": "exclude",
                "count": 100
            }
            
            headers = {
                "User-Agent": "Sample Company Name AdminContact@company.com"
            }
            
            response = requests.get(base_url, params=params, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for filing in soup.find_all("div", class_="filing"):
                news_data.append({
                    "date": filing.find("div", class_="date").text,
                    "title": filing.find("div", class_="title").text,
                    "source": "SEC EDGAR",
                    "url": "https://www.sec.gov" + filing.find("a")["href"]
                })
            
            return pd.DataFrame(news_data)
            
        except Exception as e:
            self.logger.error(f"Error fetching SEC filings: {str(e)}")
            return pd.DataFrame()

    def fetch_arxiv_papers(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch relevant research papers from arXiv
        """
        news_data = []
        
        try:
            search = arxiv.Search(
                query=f"ti:{symbol} OR abs:{symbol}",
                max_results=100,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in search.results():
                if start_date <= result.published.strftime("%Y-%m-%d") <= end_date:
                    news_data.append({
                        "date": result.published.strftime("%Y-%m-%d"),
                        "title": result.title,
                        "source": "arXiv",
                        "url": result.entry_id,
                        "abstract": result.summary
                    })
            
            return pd.DataFrame(news_data)
            
        except Exception as e:
            self.logger.error(f"Error fetching arXiv papers: {str(e)}")
            return pd.DataFrame()

    def combine_historical_sources(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Combine news from multiple sources and remove duplicates
        """
        dfs = []
        
        # Collect from all sources
        dfs.append(self.fetch_marketaux_historical(symbol, start_date, end_date))
        dfs.append(self.fetch_financial_times_archive(symbol, start_date, end_date))
        dfs.append(self.fetch_sec_filings(symbol, start_date, end_date))
        dfs.append(self.fetch_arxiv_papers(symbol, start_date, end_date))
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on title similarity
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date')
        
        # Drop exact duplicates
        combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
        
        return combined_df