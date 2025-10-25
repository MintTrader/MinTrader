"""
S3 Report Manager

Handles all S3 operations for storing and retrieving reports, logs, and summaries.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Get logger for this module
logger = logging.getLogger(__name__)


class S3ReportManager:
    """Manages report storage and retrieval in S3"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        """
        Initialize S3 client.
        
        Args:
            bucket_name: Name of the S3 bucket
            region: AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize boto3 client
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.warning(f"Could not create bucket: {create_error}")
    
    def upload_reports(self, ticker: str, date: str, reports_dir: Path) -> bool:
        """
        Upload all reports for a stock analysis to S3.
        
        Args:
            ticker: Stock ticker symbol
            date: Analysis date (YYYY-MM-DD)
            reports_dir: Local directory containing reports
            
        Returns:
            True if successful, False otherwise
        """
        try:
            reports_path = Path(reports_dir)
            if not reports_path.exists():
                logger.warning(f"Reports directory not found: {reports_dir}")
                return False
            
            # Upload all markdown files
            for report_file in reports_path.glob('*.md'):
                s3_key = f"reports/{ticker}/{date}/{report_file.name}"
                self.s3_client.upload_file(
                    str(report_file),
                    self.bucket_name,
                    s3_key
                )
            
            # Upload HTML if it exists
            html_file = reports_path.parent / 'index.html'
            if html_file.exists():
                s3_key = f"reports/{ticker}/{date}/index.html"
                self.s3_client.upload_file(
                    str(html_file),
                    self.bucket_name,
                    s3_key
                )
            
            logger.info(f"Uploaded reports for {ticker} on {date} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading reports: {e}")
            return False
    
    def upload_log(self, iteration_id: str, log_path: Path) -> bool:
        """
        Upload message_tool.log to S3.
        
        Args:
            iteration_id: Unique iteration identifier
            log_path: Local path to log file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not Path(log_path).exists():
                logger.warning(f"Log file not found: {log_path}")
                return False
            
            s3_key = f"logs/{iteration_id}/message_tool.log"
            self.s3_client.upload_file(
                str(log_path),
                self.bucket_name,
                s3_key
            )
            
            logger.info(f"Uploaded log for iteration {iteration_id} to S3")
            return True
            
        except Exception as e:
            print(f"Error uploading log: {e}")
            return False
    
    def get_last_summary(self) -> Optional[str]:
        """
        Retrieve last iteration summary from S3.
        
        Returns:
            Summary text or None if not found
        """
        try:
            s3_key = "summaries/latest.txt"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            summary = response['Body'].read().decode('utf-8')
            return summary
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return None
            logger.error(f"Error retrieving last summary: {e}")
            return None
    
    def save_summary(self, summary: str, iteration_id: str) -> bool:
        """
        Save current iteration summary to S3.
        
        Args:
            summary: Summary text
            iteration_id: Unique iteration identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save with iteration ID
            s3_key = f"summaries/{iteration_id}.txt"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=summary.encode('utf-8')
            )
            
            # Also save as latest
            latest_key = "summaries/latest.txt"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=latest_key,
                Body=summary.encode('utf-8')
            )
            
            logger.info(f"Saved summary for iteration {iteration_id} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            return False
    
    def get_position_history(self) -> Dict[str, str]:
        """
        Retrieve position entry dates from S3.
        
        Returns:
            Dictionary mapping ticker to entry date (YYYY-MM-DD)
        """
        try:
            s3_key = "positions/history.json"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            history = json.loads(response['Body'].read().decode('utf-8'))
            return history
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return {}
            logger.error(f"Error retrieving position history: {e}")
            return {}
    
    def save_position_history(self, history: Dict[str, str]) -> bool:
        """
        Save position entry dates to S3.
        
        Args:
            history: Dictionary mapping ticker to entry date
            
        Returns:
            True if successful, False otherwise
        """
        try:
            s3_key = "positions/history.json"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(history, indent=2).encode('utf-8')
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving position history: {e}")
            return False
    
    def list_reports(self, ticker: Optional[str] = None) -> List[str]:
        """
        List all reports in S3, optionally filtered by ticker.
        
        Args:
            ticker: Optional ticker to filter by
            
        Returns:
            List of S3 keys
        """
        try:
            prefix = f"reports/{ticker}/" if ticker else "reports/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            return [obj['Key'] for obj in response['Contents']]
            
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
            return []
    
    def get_analyzed_stocks_history(self, days_threshold: int = 14) -> Dict[str, List[str]]:
        """
        Fetch all stock tickers and their analysis dates from S3 reports folder.
        
        This method scans the S3 bucket's "reports/" folder structure:
        - Top-level folders are stock ticker names (e.g., AAPL, TSLA)
        - Second-level folders are dates in YYYY-MM-DD format
        
        Args:
            days_threshold: Number of days to look back (default: 14)
            
        Returns:
            Dictionary mapping ticker symbols to list of analysis dates
            Example: {'AAPL': ['2025-10-18', '2025-10-15'], 'TSLA': ['2025-10-17']}
        """
        try:
            stock_history = {}
            
            # Calculate cutoff date
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Use paginator to handle large result sets
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # List all objects in reports/ folder
            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix='reports/',
                Delimiter='/'
            ):
                # Get ticker folders (CommonPrefixes are the "directories")
                if 'CommonPrefixes' in page:
                    for prefix in page['CommonPrefixes']:
                        # Extract ticker from prefix (e.g., "reports/AAPL/" -> "AAPL")
                        ticker_path = prefix['Prefix']
                        ticker = ticker_path.rstrip('/').split('/')[-1]
                        
                        # Now get date folders for this ticker
                        date_folders = []
                        for date_page in paginator.paginate(
                            Bucket=self.bucket_name,
                            Prefix=ticker_path,
                            Delimiter='/'
                        ):
                            if 'CommonPrefixes' in date_page:
                                for date_prefix in date_page['CommonPrefixes']:
                                    # Extract date from prefix (e.g., "reports/AAPL/2025-10-18/" -> "2025-10-18")
                                    date_str = date_prefix['Prefix'].rstrip('/').split('/')[-1]
                                    
                                    # Validate date format and check if within threshold
                                    try:
                                        analysis_date = datetime.strptime(date_str, "%Y-%m-%d")
                                        if analysis_date >= cutoff_date:
                                            date_folders.append(date_str)
                                    except ValueError:
                                        # Skip invalid date formats
                                        continue
                        
                        # Add ticker to history if it has analysis dates
                        if date_folders:
                            # Sort dates in descending order (most recent first)
                            date_folders.sort(reverse=True)
                            stock_history[ticker] = date_folders
            
            return stock_history
            
        except Exception as e:
            logger.error(f"Error fetching analyzed stocks history from S3: {e}")
            return {}
    
    def get_report_from_s3(self, ticker: str, report_type: str, date: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a specific analysis report from S3.
        
        Args:
            ticker: Stock ticker symbol
            report_type: Type of report (e.g., 'final_trade_decision', 'investment_plan', 'market_report')
            date: Optional specific date (YYYY-MM-DD). If None, gets the most recent report.
            
        Returns:
            Report content as string, or None if not found
        """
        try:
            # If no date specified, find the most recent date for this ticker
            if date is None:
                # List all date folders for this ticker
                paginator = self.s3_client.get_paginator('list_objects_v2')
                date_folders = []
                
                for page in paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=f'reports/{ticker}/',
                    Delimiter='/'
                ):
                    if 'CommonPrefixes' in page:
                        for prefix in page['CommonPrefixes']:
                            date_str = prefix['Prefix'].rstrip('/').split('/')[-1]
                            try:
                                # Validate date format
                                datetime.strptime(date_str, "%Y-%m-%d")
                                date_folders.append(date_str)
                            except ValueError:
                                continue
                
                if not date_folders:
                    logger.info(f"No analysis reports found for {ticker}")
                    return None
                
                # Get the most recent date
                date_folders.sort(reverse=True)
                date = date_folders[0]
            
            # Fetch the report
            s3_key = f"reports/{ticker}/{date}/{report_type}.md"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            content = response['Body'].read().decode('utf-8')
            return content
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.info(f"Report '{report_type}' not found for {ticker} on {date}")
                return None
            logger.error(f"Error retrieving report for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving report for {ticker}: {e}")
            return None
    
    def get_stock_analysis_decision(self, ticker: str, date: str) -> str:
        """
        Retrieve the trading decision for a specific stock analysis from S3.
        
        Args:
            ticker: Stock ticker symbol
            date: Analysis date (YYYY-MM-DD)
            
        Returns:
            Decision string ("BUY", "SELL", "HOLD", or "UNKNOWN")
        """
        try:
            s3_key = f"reports/{ticker}/{date}/final_trade_decision.md"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            content = response['Body'].read().decode('utf-8')
            content_upper = content.upper()  # Case-insensitive matching
            
            # Extract decision from content - check multiple patterns
            # Common patterns: "Recommendation: Buy", "Decision: BUY", "Final Recommendation: Hold"
            if any(pattern in content_upper for pattern in [
                "RECOMMENDATION: BUY", "RECOMMENDATION:**BUY**", "RECOMMENDATION: **BUY**",
                "DECISION: BUY", "DECISION:**BUY**", "DECISION: **BUY**"
            ]):
                return "BUY"
            elif any(pattern in content_upper for pattern in [
                "RECOMMENDATION: SELL", "RECOMMENDATION:**SELL**", "RECOMMENDATION: **SELL**",
                "DECISION: SELL", "DECISION:**SELL**", "DECISION: **SELL**"
            ]):
                return "SELL"
            elif any(pattern in content_upper for pattern in [
                "RECOMMENDATION: HOLD", "RECOMMENDATION:**HOLD**", "RECOMMENDATION: **HOLD**",
                "DECISION: HOLD", "DECISION:**HOLD**", "DECISION: **HOLD**",
                "FINAL RECOMMENDATION:**HOLD**", "FINAL RECOMMENDATION: **HOLD**"
            ]):
                return "HOLD"
            else:
                # Log the beginning of content for debugging
                logger.debug(f"Could not determine decision for {ticker} on {date}. Content preview: {content[:200]}...")
                return "UNKNOWN"
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.debug(f"Decision file not found for {ticker} on {date}")
                return "UNKNOWN"
            logger.error(f"Error retrieving decision for {ticker} on {date}: {e}")
            return "UNKNOWN"
        except Exception as e:
            logger.error(f"Error retrieving decision for {ticker} on {date}: {e}")
            return "UNKNOWN"

