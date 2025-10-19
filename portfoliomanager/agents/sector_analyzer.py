"""
Sector Analysis Agent

Analyzes current portfolio positions and recommends which sectors
to invest in based on diversification goals.
"""

from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import json


def create_sector_analyzer(llm):
    """
    Create a sector analyzer agent that recommends which sectors to invest in.
    
    Args:
        llm: Language model instance
        
    Returns:
        Function that analyzes sectors
    """
    
    system_message = """You are a portfolio diversification expert. Your job is to analyze the current portfolio positions and recommend which sectors we should focus on for new investments.

**YOUR ANALYSIS PROCESS:**
1. Review current portfolio positions and their sectors
2. Calculate current sector exposure (what % of portfolio is in each sector)
3. Consider the portfolio's concentration risk
4. Identify sectors that are:
   - Underrepresented or missing (good for diversification)
   - Have strong market fundamentals
   - Align with medium-term growth strategy
5. Recommend 1-3 sectors to focus on for new investments

**CRITICAL RULES:**
- Maximum 30% of portfolio should be in any single sector (check current exposure)
- If a sector already has 25%+ exposure, DO NOT recommend it
- Prefer sectors with 0-15% exposure for best diversification
- If portfolio is well-diversified across 5+ sectors at 15-20% each, return EMPTY list
- Quality over quantity - only recommend sectors you have high conviction on

**OUTPUT FORMAT:**
You must respond with a valid JSON object in this exact format:
{{
    "recommended_sectors": ["Technology", "Healthcare"],
    "current_exposure": {{
        "Technology": 25.5,
        "Healthcare": 10.2,
        "Finance": 0.0,
        "Consumer": 0.0,
        "Energy": 0.0,
        "Industrial": 0.0,
        "Real Estate": 0.0,
        "Materials": 0.0,
        "Utilities": 0.0,
        "Communication Services": 0.0
    }},
    "reasoning": "Brief explanation of why these sectors",
    "should_invest": true
}}

If portfolio is well-diversified and no new investments needed:
{{
    "recommended_sectors": [],
    "current_exposure": {{...all sectors with their percentages...}},
    "reasoning": "Portfolio is well-diversified across sectors",
    "should_invest": false
}}

**IMPORTANT:** 
- Only return valid JSON, no other text
- ALWAYS include ALL major sectors in current_exposure, even if they have 0% exposure
- Sector names must be standard categories: Technology, Healthcare, Finance, Energy, Consumer, Industrial, Real Estate, Materials, Utilities, Communication Services
- If you're unsure of a company's sector, use "Unknown"
"""
    
    def analyze_sectors(positions: List[Dict[str, Any]], max_concentration_pct: float = 30) -> Dict[str, Any]:
        """
        Analyze current positions and recommend sectors.
        
        Args:
            positions: List of current position dictionaries
            max_concentration_pct: Maximum allowed sector concentration
            
        Returns:
            Dictionary with sector recommendations
        """
        # Format positions for the prompt
        if not positions:
            positions_text = "No current positions - portfolio is empty"
        else:
            positions_text = "Current Positions:\n"
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for pos in positions:
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('qty', 0)
                market_value = pos.get('market_value', 0)
                pct = (market_value / total_value * 100) if total_value > 0 else 0
                positions_text += f"- {symbol}: {qty} shares, ${market_value:,.2f} ({pct:.1f}% of portfolio)\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", f"{positions_text}\n\nAnalyze the portfolio and recommend which sectors we should focus on for new investments. Maximum sector concentration allowed: {max_concentration_pct}%\n\nRespond with ONLY the JSON object, no other text.")
        ])
        
        chain = prompt | llm
        result = chain.invoke({})
        
        # Parse the JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            content = result.content.strip()
            
            # Find JSON object in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                # Fallback if no JSON found
                analysis = {
                    "recommended_sectors": [],
                    "current_exposure": {},
                    "reasoning": "Failed to parse LLM response",
                    "should_invest": False
                }
            
            return analysis
            
        except json.JSONDecodeError as e:
            # If parsing fails, return safe default
            return {
                "recommended_sectors": [],
                "current_exposure": {},
                "reasoning": f"Error parsing response: {e}",
                "should_invest": False
            }
    
    return analyze_sectors

