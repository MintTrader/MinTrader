"""
S3 Report Manager

Handles all S3 operations for storing and retrieving reports, logs, and summaries.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import boto3
from botocore.exceptions import ClientError


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
                    print(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_error:
                    print(f"Warning: Could not create bucket: {create_error}")
    
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
                print(f"Reports directory not found: {reports_dir}")
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
            
            print(f"Uploaded reports for {ticker} on {date} to S3")
            return True
            
        except Exception as e:
            print(f"Error uploading reports: {e}")
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
                print(f"Log file not found: {log_path}")
                return False
            
            s3_key = f"logs/{iteration_id}/message_tool.log"
            self.s3_client.upload_file(
                str(log_path),
                self.bucket_name,
                s3_key
            )
            
            print(f"Uploaded log for iteration {iteration_id} to S3")
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
            print(f"Error retrieving last summary: {e}")
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
            
            print(f"Saved summary for iteration {iteration_id} to S3")
            return True
            
        except Exception as e:
            print(f"Error saving summary: {e}")
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
            print(f"Error retrieving position history: {e}")
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
            print(f"Error saving position history: {e}")
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
            print(f"Error listing reports: {e}")
            return []

