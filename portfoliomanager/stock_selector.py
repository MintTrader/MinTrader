"""
Stock Selector Module

Handles stock selection logic, market context building, and decision-making.
"""

import json
import re
from typing import List, Dict, Any


class StockSelector:
    """Handles stock selection and market context analysis"""
    
    def __init__(self, llm, logger, max_analyses: int = 3):
        """
        Initialize stock selector.
        
        Args:
            llm: LLM instance for decision-making
            logger: PortfolioLogger instance
            max_analyses: Maximum number of analyses allowed
        """
        self.llm = llm
        self.logger = logger
        self.max_analyses = max_analyses
    
    def build_market_search_query(self, positions, open_orders, recent_analysis) -> str:
        """
        Build a comprehensive web search query covering all portfolio needs.
        
        Args:
            positions: Current positions
            open_orders: Open orders
            recent_analysis: Recently analyzed stocks
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # General market conditions
        query_parts.append("Current stock market conditions and major news")
        
        # Check on existing positions
        if positions:
            position_tickers = [p.get('symbol', '') for p in positions[:3]]
            if position_tickers:
                query_parts.append(f"Recent news on {', '.join(position_tickers)}")
        
        # Check on pending orders
        if open_orders:
            order_tickers = list(set([o.get('symbol', '') for o in open_orders[:2]]))
            if order_tickers:
                query_parts.append(f"Price movement and news on {', '.join(order_tickers)}")
        
        # If we need new opportunities
        if not positions or len(positions) < 5:
            query_parts.append(
                "List specific stock tickers: 5-10 stocks with strong growth potential "
                "and catalysts - undervalued companies with improving fundamentals, "
                "upcoming catalysts, or emerging market opportunities"
            )
        
        return " | ".join(query_parts)
    
    def decide_stocks_to_analyze(
        self, account, positions, open_orders, market_context, 
        recent_analysis, last_summary
    ) -> List[str]:
        """
        Use LLM to decide which 0-3 stocks to analyze.
        
        Args:
            account: Account information
            positions: Current positions
            open_orders: Open orders
            market_context: Web search results
            recent_analysis: Recently analyzed stocks
            last_summary: Previous iteration summary
            
        Returns:
            List of 0-3 ticker symbols to analyze
        """
        # Build exclusion list
        stocks_to_exclude, recently_analyzed_tickers = self._build_exclusion_list(recent_analysis)
        
        # Build prompt
        prompt = self._build_selection_prompt(
            account, positions, open_orders, market_context,
            recent_analysis, last_summary, stocks_to_exclude, recently_analyzed_tickers
        )
        
        try:
            response = self.llm.invoke(prompt)
            content = str(response.content).strip() if response.content else ""
            
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                selected_stocks = json.loads(json_str)
                
                # Validate and limit
                selected_stocks = [s.upper() for s in selected_stocks if isinstance(s, str)]
                
                # Filter out excluded stocks
                original_selection = selected_stocks.copy()
                selected_stocks = [s for s in selected_stocks if s not in stocks_to_exclude]
                
                # Log filtering
                if len(original_selection) > len(selected_stocks):
                    excluded = [s for s in original_selection if s in stocks_to_exclude]
                    self.logger.log_system(
                        f"⚠️  LLM ignored exclusion list and selected: {excluded}. "
                        f"Filtered them out."
                    )
                elif stocks_to_exclude:
                    self.logger.log_system(
                        f"✅ LLM successfully avoided {len(stocks_to_exclude)} recently analyzed stocks"
                    )
                
                # Limit to max analyses
                selected_stocks = selected_stocks[:self.max_analyses]
                
                if selected_stocks:
                    self.logger.log_system(f"LLM selected {len(selected_stocks)} stocks: {selected_stocks}")
                else:
                    self.logger.log_system("LLM selected 0 stocks - portfolio well-positioned or no opportunities")
                
                return selected_stocks
            else:
                self.logger.log_system("⚠️  Failed to parse LLM response")
                return []
                
        except Exception as e:
            self.logger.log_system(f"⚠️  Error deciding stocks: {e}")
            return []
    
    def _build_exclusion_list(self, recent_analysis: Dict[str, Any]) -> tuple:
        """
        Build list of stocks to exclude from analysis.
        
        Returns:
            Tuple of (stocks_to_exclude set, recently_analyzed_tickers set)
        """
        stocks_to_exclude = set()
        recently_analyzed_tickers = set()
        
        if recent_analysis.get('recently_analyzed'):
            for ra in recent_analysis['recently_analyzed']:
                ticker = ra['ticker']
                days_ago = ra['days_ago']
                
                if days_ago < 14:
                    recently_analyzed_tickers.add(ticker)
                
                if days_ago < 3:
                    stocks_to_exclude.add(ticker)
        
        return stocks_to_exclude, recently_analyzed_tickers
    
    def _build_selection_prompt(
        self, account, positions, open_orders, market_context,
        recent_analysis, last_summary, stocks_to_exclude, recently_analyzed_tickers
    ) -> str:
        """Build the stock selection prompt for LLM."""
        context_parts = []
        
        # FIRST: Show complete analysis history with dates (MOST IMPORTANT)
        context_parts.append("=" * 80)
        context_parts.append("📊 COMPLETE STOCK ANALYSIS HISTORY (Past 14 Days)")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        recently_analyzed = recent_analysis.get('recently_analyzed', [])
        if recently_analyzed:
            context_parts.append("Below is EVERY stock we've analyzed recently with exact dates.")
            context_parts.append("Use this to make informed decisions about what to analyze next:")
            context_parts.append("")
            
            # Group by freshness
            very_recent = []  # 0-2 days
            recent = []       # 3-6 days
            older = []        # 7-14 days
            
            for item in recently_analyzed:
                ticker = item['ticker']
                date = item['date']
                days_ago = item['days_ago']
                source = item.get('source', 'Unknown')
                
                entry = f"  • {ticker}: Analyzed on {date} ({days_ago} days ago, from {source})"
                
                if days_ago <= 2:
                    very_recent.append(entry)
                elif days_ago <= 6:
                    recent.append(entry)
                else:
                    older.append(entry)
            
            if very_recent:
                context_parts.append("🔴 VERY RECENT (0-2 days) - DO NOT RE-ANALYZE:")
                context_parts.extend(very_recent)
                context_parts.append("")
            
            if recent:
                context_parts.append("🟡 RECENT (3-6 days) - Only re-analyze if major news:")
                context_parts.extend(recent)
                context_parts.append("")
            
            if older:
                context_parts.append("🟢 OLDER (7-14 days) - OK to re-analyze if needed:")
                context_parts.extend(older)
                context_parts.append("")
            
            context_parts.append(f"Total stocks analyzed in past 14 days: {len(recently_analyzed)}")
            context_parts.append("")
            context_parts.append("💡 IMPORTANT: Check if a stock appears above BEFORE selecting it!")
            context_parts.append("   - Stocks in 🔴 VERY RECENT: NEVER select")
            context_parts.append("   - Stocks in 🟡 RECENT: Only if major news/catalyst")
            context_parts.append("   - Stocks in 🟢 OLDER: Can re-analyze if relevant")
            context_parts.append("   - Stocks NOT listed above: Fresh analysis available!")
        else:
            context_parts.append("✅ No stocks analyzed in past 14 days - all stocks available!")
        
        context_parts.append("")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        # SECOND: Simplified exclusion reminder
        if stocks_to_exclude:
            excluded_tickers_str = ", ".join(sorted(stocks_to_exclude))
            context_parts.append("⚠️  QUICK REMINDER - DO NOT SELECT: " + excluded_tickers_str)
            context_parts.append("   (These are in the 🔴 VERY RECENT category above)")
        else:
            context_parts.append("✅ No hard exclusions - use the analysis history above to guide your choices")
        
        context_parts.append("")
        
        # Add schedule information
        context_parts.append("=== YOUR SCHEDULE (For Temporal Context) ===")
        context_parts.append("You run 3 times per day, Monday-Friday:")
        context_parts.append("  • Morning: Market open (9:30 AM ET)")
        context_parts.append("  • Midday: Market midpoint (12:30 PM ET)")
        context_parts.append("  • Evening: 30 min before close (3:30 PM ET)")
        context_parts.append("  • Weekends: No runs (market closed)")
        context_parts.append("")
        context_parts.append("This means you'll see the same stocks 2-3 times per day if needed.")
        context_parts.append("Consider market hours when deciding what to analyze.")
        context_parts.append("")
        
        # Previous iteration context
        if last_summary and "No previous iteration" not in last_summary:
            context_parts.append("=== PREVIOUS ITERATION SUMMARY ===")
            summary_lines = last_summary.split('\n')
            for line in summary_lines[:10]:
                if line.strip():
                    context_parts.append(line)
            context_parts.append("")
        
        # Account info
        context_parts.append("=== CURRENT ACCOUNT STATE ===")
        context_parts.append(f"Cash: ${account.get('cash', 0):,.2f}")
        context_parts.append(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        
        # Positions - show with analysis dates
        if positions:
            context_parts.append(f"\nCurrent Positions ({len(positions)}):")
            
            # Build a lookup map for analysis dates
            analysis_date_map = {}
            for item in recently_analyzed:
                analysis_date_map[item['ticker']] = {
                    'date': item['date'],
                    'days_ago': item['days_ago']
                }
            
            for p in positions:
                ticker = p['symbol']
                pnl_pct = (p.get('unrealized_plpc', 0) * 100)
                
                # Show analysis date if available
                if ticker in analysis_date_map:
                    date_info = analysis_date_map[ticker]
                    days = date_info['days_ago']
                    date = date_info['date']
                    
                    if days <= 2:
                        status = f"🔴 Last analyzed: {date} ({days}d ago) - DO NOT SELECT"
                    elif days <= 6:
                        status = f"🟡 Last analyzed: {date} ({days}d ago) - Only if major news"
                    else:
                        status = f"🟢 Last analyzed: {date} ({days}d ago) - OK to re-analyze"
                else:
                    status = "⚠️  NEEDS ANALYSIS - No recent analysis found!"
                
                context_parts.append(
                    f"  - {ticker}: {p['qty']} shares, ${p.get('market_value', 0):,.2f} "
                    f"({pnl_pct:+.1f}%) | {status}"
                )
        else:
            context_parts.append("\nNo current positions")
        
        # Open orders
        if open_orders:
            context_parts.append(f"\nPending Orders ({len(open_orders)}):")
            for o in open_orders:
                context_parts.append(
                    f"  - {o['symbol']}: {o['side']} {o['qty']} @ ${o.get('limit_price', 'market')}"
                )
        
        # Market context
        context_parts.append(f"\n=== MARKET CONTEXT & NEW OPPORTUNITIES ===")
        context_parts.append("Read this carefully for NEW stocks to analyze:")
        context_parts.append("")
        context_parts.append(market_context[:3000])
        context_parts.append("")
        
        # Extract potential tickers from market context
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', market_context[:3000])
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'WITH', 
            'FROM', 'THIS', 'THAT', 'NYSE', 'NASDAQ', 'ETF', 'IPO', 'CEO', 'CFO'
        }
        potential_tickers = [t for t in potential_tickers if t not in common_words and len(t) <= 5]
        
        if potential_tickers:
            unique_tickers = list(dict.fromkeys(potential_tickers))[:20]
            context_parts.append(f"💡 POTENTIAL TICKERS EXTRACTED: {', '.join(unique_tickers)}")
            context_parts.append("   (Note: May need manual ticker lookup for company names like 'General Motors' → GM)")
        
        # Add explicit reminder about new opportunities
        context_parts.append("")
        context_parts.append("🔍 LOOK FOR NEW OPPORTUNITIES IN THE MARKET CONTEXT ABOVE:")
        context_parts.append("   - Companies mentioned with positive earnings (e.g., 'General Motors' → ticker: GM)")
        context_parts.append("   - Stocks with strong momentum or news catalysts")
        context_parts.append("   - Sectors showing strength")
        context_parts.append("   - Convert company names to ticker symbols if needed")
        context_parts.append("   - These are FRESH candidates - not in your analysis history!")
        
        # Build instruction section
        prompt = f"""Based on the portfolio state and market context, select 0-3 stocks to analyze for trading.

{chr(10).join(context_parts)}

================================================================================
YOUR TASK: SELECT 0-3 STOCKS FOR ANALYSIS
================================================================================

🔍 STEP 1: CHECK THE ANALYSIS HISTORY TABLE AT THE TOP
   - Review the "📊 COMPLETE STOCK ANALYSIS HISTORY" section above
   - Note which stocks were analyzed and when
   - 🔴 VERY RECENT (0-2 days): NEVER re-analyze these
   - 🟡 RECENT (3-6 days): Only if major news
   - 🟢 OLDER (7-14 days): Can re-analyze if needed
   - Any stock NOT in the table: Fresh analysis available

🎯 STEP 2: APPLY SELECTION STRATEGY

RULES:
1. ✅ Maximum 3 stocks total
2. 🚫 Check the analysis history BEFORE selecting any stock
3. 💡 You can still TRADE recently analyzed stocks using read_historical_report

SELECTION STRATEGY:

🎯 YOUR PRIMARY JOB: FIND NEW OPPORTUNITIES!

You should ACTIVELY look for new stocks to analyze, not just maintain existing positions.

1️⃣ NEW INVESTMENT OPPORTUNITIES (PRIORITIZE THIS!):
   ✅ Read the "MARKET CONTEXT & NEW OPPORTUNITIES" section carefully
   ✅ Look for companies with:
      - Strong earnings reports (e.g., "General Motors" with positive earnings → ticker: GM)
      - Momentum and positive news (e.g., "RTX" mentioned positively → ticker: RTX)
      - Growth catalysts (new products, partnerships, market expansion)
      - Sector strength (e.g., if tech sector is strong, find tech stocks)
   ✅ Convert company names to ticker symbols (e.g., "Coca-Cola" → KO, "Danaher" → DHR)
   ✅ IMPORTANT: Cross-check with the analysis history table to avoid recently analyzed stocks
   
   💡 The market context is specifically designed to surface NEW opportunities!
   💡 Don't ignore it - it's your primary source for finding what to analyze!

2️⃣ EXISTING POSITIONS marked "⚠️ NEEDS ANALYSIS" (Secondary Priority):
   - These haven't been analyzed recently (check the history table!)
   - Prioritize positions with large gains/losses
   - Good candidates if they're NOT in the 🔴 VERY RECENT category

SUGGESTED APPROACH:
• READ the market context section thoroughly - companies mentioned there are candidates!
• If market context mentions companies with positive news → SELECT THEM (convert names to tickers)
• If 1-2 positions need analysis → Mix: 1 new opportunity + 1-2 existing positions
• If all positions recently analyzed → SELECT 2-3 NEW opportunities from market context
• ONLY select 0 stocks if:
  - Market context has NO actionable opportunities AND
  - All positions were analyzed very recently (0-2 days) AND
  - Portfolio is well-balanced with no concerns

⚠️  IMPORTANT: Selecting 0 stocks means you're passing up ALL opportunities in the market context!
    Make sure that's intentional - usually you should find 1-3 stocks to analyze.

📋 EXAMPLES OF GOOD DECISION-MAKING:

Example 1 - NEW OPPORTUNITIES FROM MARKET CONTEXT:
✅ Market context says: "Strong earnings from General Motors, RTX, Danaher, Coca-Cola"
✅ Decision: "I'll select GM, RTX, and DHR - all have positive news and aren't in my analysis history"
✅ Result: ["GM", "RTX", "DHR"] - 3 new opportunities to analyze!

Example 2 - MIXED APPROACH:
✅ Market context mentions RTX with positive news (not in history)
✅ I have GOOGL position, last analyzed 8 days ago (🟢)
✅ Decision: "I'll analyze RTX (new opportunity) and GOOGL (existing position needing refresh)"
✅ Result: ["RTX", "GOOGL"] - 1 new + 1 existing

Example 3 - CONSERVATIVE (Only when justified):
❌ BAD: "All my positions were analyzed yesterday, so I'll select []"
✅ GOOD: "My positions were analyzed yesterday BUT market context shows GM and KO with strong earnings → ["GM", "KO"]"

Example 4 - TRULY NO OPPORTUNITIES (Rare):
✅ "Market context shows general market update with no specific stock opportunities"
✅ "All positions analyzed 0-1 days ago"
✅ "No concerning portfolio issues"
✅ Decision: [] - Genuinely nothing to do this iteration

Respond with ONLY a JSON list of ticker symbols.
Examples:
- ["NVDA", "TSLA"]: Two fresh stocks to analyze
- ["AAPL"]: One stock (checked history first)
- []: No stocks (portfolio well-positioned or no fresh candidates)
"""
        
        return prompt

