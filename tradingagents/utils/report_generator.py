"""
Unified HTML Report Generator for Trading Agents Analysis
Converts all markdown reports into a single, beautifully formatted HTML page
"""

from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import markdown


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
        
        html {{
            scroll-behavior: smooth;
        }}
        
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
        ('market_report.md', 'Market Analysis', 'Technical indicators and market trends'),
        ('news_report.md', 'News Analysis', 'Recent news and market events'),
        ('sentiment_report.md', 'Sentiment Analysis', 'Social media and public sentiment'),
        ('fundamentals_report.md', 'Fundamental Analysis', 'Financial statements and company fundamentals'),
        ('investment_plan.md', 'Investment Plan', 'Research manager recommendations'),
        ('trader_investment_plan.md', 'Trader Investment Plan', 'Risk analysis and final decision'),
        ('final_trade_decision.md', 'Final Trade Decision', 'Portfolio manager final decision'),
    ]

    @classmethod
    def generate_for_analysis(cls, results_dir: Path) -> bool:
        """
        Generate HTML report for a completed analysis
        
        Args:
            results_dir: Path to the results directory (e.g., results/MNDY/2022-01-01)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results_dir = Path(results_dir)
            reports_dir = results_dir / "reports"
            
            if not reports_dir.exists():
                return False
            
            # Extract ticker and date from path
            ticker = results_dir.parent.name
            analysis_date = results_dir.name
            
            # Find all reports
            reports = []
            for filename, title, description in cls.REPORT_ORDER:
                filepath = reports_dir / filename
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    reports.append((filename, title, description, content))
            
            if not reports:
                return False
            
            # Generate navigation
            nav_items = []
            for filename, title, _, _ in reports:
                section_id = filename.replace('.md', '')
                nav_items.append(f'<a href="#{section_id}">{title}</a>')
            navigation = '\n            '.join(nav_items)
            
            # Generate sections
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
            
            sections_html = '\n'.join(sections)
            
            # Generate final HTML
            html = cls.HTML_TEMPLATE.format(
                title=f"{ticker} Analysis - {analysis_date}",
                ticker=ticker,
                analysis_date=analysis_date,
                generated_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report_count=len(reports),
                navigation=navigation,
                sections=sections_html,
                year=datetime.now().year
            )
            
            # Save to index.html
            output_path = results_dir / "index.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            return True
            
        except Exception:
            return False

