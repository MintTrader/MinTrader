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
        
        # Check if credentials are available
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    
        
        # Initialize boto3 client
        self.s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret
        )
        
        logger.info(f"S3 client initialized for bucket '{bucket_name}' in region '{region}'")
        
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
            logger.info(f"Attempting to fetch summary from s3://{self.bucket_name}/{s3_key}")
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            summary = response['Body'].read().decode('utf-8')
            logger.info("Successfully retrieved last summary from S3")
            return summary
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.info("No previous summary found (this is OK for first run)")
                return None
            elif error_code == 'AccessDenied':
                logger.error(f"Access Denied to S3 bucket '{self.bucket_name}'")
                logger.error("Please verify:")
                logger.error("  1. AWS credentials are correct and loaded")
                logger.error("  2. IAM user has s3:GetObject permission")
                logger.error(f"  3. Bucket '{self.bucket_name}' exists in region '{self.region}'")
                logger.error(f"  4. Bucket policy allows access from your IAM user")
            else:
                logger.error(f"Error retrieving last summary (code: {error_code}): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving last summary: {e}")
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
