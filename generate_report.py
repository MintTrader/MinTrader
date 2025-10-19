#!/usr/bin/env python3
"""
Unified HTML Report Generator for Trading Agents Analysis
Converts all markdown reports into a single, beautifully formatted HTML page
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import markdown
from typing import Dict, List, Tuple
import argparse


class ReportGenerator:
    """Generate unified HTML reports from markdown files"""
    
    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .metadata {{
            font-size: 1.1em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        
        .nav {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 2px solid #e9ecef;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .nav a {{
            text-decoration: none;
            color: #667eea;
            padding: 10px 20px;
            border-radius: 6px;
            transition: all 0.3s ease;
            font-weight: 500;
            background: white;
            border: 2px solid #667eea;
        }}
        
        .nav a:hover {{
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            margin: -10px -10px 30px -10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }}
        
        .section-header h2 {{
            font-size: 2em;
            font-weight: 600;
        }}
        
        .section-header .report-type {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        
        .report-content {{
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        .report-content h1,
        .report-content h2,
        .report-content h3 {{
            color: #667eea;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
        }}
        
        .report-content h1 {{
            font-size: 2em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .report-content h2 {{
            font-size: 1.5em;
        }}
        
        .report-content h3 {{
            font-size: 1.2em;
        }}
        
        .report-content p {{
            margin: 1em 0;
            text-align: justify;
        }}
        
        .report-content ul,
        .report-content ol {{
            margin: 1em 0 1em 2em;
        }}
        
        .report-content li {{
            margin: 0.5em 0;
        }}
        
        .report-content table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .report-content th,
        .report-content td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .report-content th {{
            background: #667eea;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }}
        
        .report-content tr:last-child td {{
            border-bottom: none;
        }}
        
        .report-content tr:hover {{
            background: #f8f9fa;
        }}
        
        .report-content code {{
            background: #f1f3f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }}
        
        .report-content pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1.5em 0;
        }}
        
        .report-content pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        
        .report-content blockquote {{
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin: 1.5em 0;
            color: #666;
            font-style: italic;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 2px solid #e9ecef;
        }}
        
        .footer p {{
            margin: 5px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 5px;
        }}
        
        .badge-success {{
            background: #28a745;
            color: white;
        }}
        
        .badge-warning {{
            background: #ffc107;
            color: #212529;
        }}
        
        .badge-danger {{
            background: #dc3545;
            color: white;
        }}
        
        .badge-info {{
            background: #17a2b8;
            color: white;
        }}
        
        .alert {{
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}
        
        .alert-info {{
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }}
        
        .alert-danger {{
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .nav {{
                flex-direction: column;
            }}
            
            .nav a {{
                width: 100%;
                text-align: center;
            }}
        }}
        
        /* Smooth scrolling */
        html {{
            scroll-behavior: smooth;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
            
            .nav {{
                display: none;
            }}
            
            .section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{ticker} Trading Analysis Report</h1>
            <div class="metadata">
                <p><strong>Analysis Date:</strong> {analysis_date}</p>
                <p><strong>Generated:</strong> {generated_date}</p>
                <p><strong>Reports Included:</strong> {report_count}</p>
            </div>
        </div>
        
        <nav class="nav">
            {navigation}
        </nav>
        
        <div class="content">
            {sections}
        </div>
        
        <div class="footer">
            <p><strong>MinTrader - AI-Powered Trading Analysis</strong></p>
            <p>Generated by Trading Agents Multi-Agent System</p>
            <p>&copy; {year} MinTrader. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""

    REPORT_ORDER = [
        ('final_trade_decision.md', 'Final Trade Decision', 'Portfolio manager final decision'),
        ('market_report.md', 'Market Analysis', 'Technical indicators and market trends'),
        ('news_report.md', 'News Analysis', 'Recent news and market events'),
        ('sentiment_report.md', 'Sentiment Analysis', 'Social media and public sentiment'),
        ('fundamentals_report.md', 'Fundamental Analysis', 'Financial statements and company fundamentals'),
        ('investment_plan.md', 'Investment Plan', 'Research manager recommendations'),
        ('trader_investment_plan.md', 'Trader Investment Plan', 'Risk analysis and final decision'),
    ]

    def __init__(self, results_dir: Path):
        """Initialize the report generator
        
        Args:
            results_dir: Path to the results directory (e.g., results/MNDY/2022-01-01)
        """
        self.results_dir = Path(results_dir)
        self.reports_dir = self.results_dir / "reports"
        
        if not self.reports_dir.exists():
            raise ValueError(f"Reports directory not found: {self.reports_dir}")
        
        # Extract ticker and date from path
        self.ticker = self.results_dir.parent.name
        self.analysis_date = self.results_dir.name
    
    def find_reports(self) -> List[Tuple[str, str, str, str]]:
        """Find all markdown reports in the reports directory
        
        Returns:
            List of tuples (filename, title, description, content)
        """
        reports = []
        
        for filename, title, description in self.REPORT_ORDER:
            filepath = self.reports_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                reports.append((filename, title, description, content))
        
        return reports
    
    def generate_navigation(self, reports: List[Tuple[str, str, str, str]]) -> str:
        """Generate navigation links
        
        Args:
            reports: List of report tuples
            
        Returns:
            HTML navigation string
        """
        nav_items = []
        for filename, title, _, _ in reports:
            section_id = filename.replace('.md', '')
            nav_items.append(f'<a href="#{section_id}">{title}</a>')
        
        return '\n            '.join(nav_items)
    
    def generate_sections(self, reports: List[Tuple[str, str, str, str]]) -> str:
        """Generate HTML sections from reports
        
        Args:
            reports: List of report tuples
            
        Returns:
            HTML sections string
        """
        sections = []
        
        for filename, title, description, content in reports:
            section_id = filename.replace('.md', '')
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                content,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            section_html = f"""
            <section id="{section_id}" class="section">
                <div class="section-header">
                    <h2>{title}</h2>
                    <div class="report-type">{description}</div>
                </div>
                <div class="report-content">
                    {html_content}
                </div>
            </section>"""
            
            sections.append(section_html)
        
        return '\n'.join(sections)
    
    def generate_html(self, output_path: Path = None) -> str:
        """Generate the unified HTML report
        
        Args:
            output_path: Optional path to save the HTML file
            
        Returns:
            HTML string
        """
        reports = self.find_reports()
        
        if not reports:
            raise ValueError(f"No markdown reports found in {self.reports_dir}")
        
        navigation = self.generate_navigation(reports)
        sections = self.generate_sections(reports)
        
        html = self.HTML_TEMPLATE.format(
            title=f"{self.ticker} Analysis - {self.analysis_date}",
            ticker=self.ticker,
            analysis_date=self.analysis_date,
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_count=len(reports),
            navigation=navigation,
            sections=sections,
            year=datetime.now().year
        )
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ HTML report generated: {output_path}")
        
        return html
    
    @classmethod
    def generate_all_reports(cls, results_base_dir: Path = None):
        """Generate HTML reports for all analysis runs
        
        Args:
            results_base_dir: Base results directory (default: ./results)
        """
        if results_base_dir is None:
            results_base_dir = Path(__file__).parent / "results"
        else:
            results_base_dir = Path(results_base_dir)
        
        if not results_base_dir.exists():
            print(f"❌ Results directory not found: {results_base_dir}")
            return
        
        generated_count = 0
        
        # Iterate through ticker directories
        for ticker_dir in results_base_dir.iterdir():
            if not ticker_dir.is_dir() or ticker_dir.name.startswith('.'):
                continue
            
            # Iterate through date directories
            for date_dir in ticker_dir.iterdir():
                if not date_dir.is_dir() or date_dir.name.startswith('.'):
                    continue
                
                reports_dir = date_dir / "reports"
                if not reports_dir.exists():
                    continue
                
                try:
                    generator = cls(date_dir)
                    output_file = date_dir / "unified_report.html"
                    generator.generate_html(output_file)
                    generated_count += 1
                except Exception as e:
                    print(f"⚠️  Error generating report for {date_dir}: {e}")
        
        print(f"\n✨ Generated {generated_count} HTML reports")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate unified HTML reports from trading analysis markdown files'
    )
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to specific analysis directory (e.g., results/MNDY/2022-01-01) or base results directory'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate reports for all analysis runs'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output file path (default: unified_report.html in analysis directory)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.all or (args.path and Path(args.path).name == 'results'):
            # Generate all reports
            base_dir = Path(args.path) if args.path else None
            ReportGenerator.generate_all_reports(base_dir)
        elif args.path:
            # Generate single report
            path = Path(args.path)
            if not path.exists():
                print(f"❌ Path not found: {path}")
                sys.exit(1)
            
            generator = ReportGenerator(path)
            output_path = Path(args.output) if args.output else path / "unified_report.html"
            generator.generate_html(output_path)
        else:
            # Default: generate all reports
            ReportGenerator.generate_all_reports()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

