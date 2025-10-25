import chromadb
from chromadb.config import Settings


class FinancialSituationMemory:
    def __init__(self, name, config):
        self.config = config
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        
        # Determine if we should use OpenAI or ChromaDB's built-in embeddings
        llm_provider = config.get("llm_provider", "openai")
        self.use_openai_embeddings = llm_provider not in ["ollama"]
        
        if self.use_openai_embeddings:
            # Use OpenAI embeddings (requires OPENAI_API_KEY)
            from openai import OpenAI
            self.embedding_model = "text-embedding-3-small"
            self.client = OpenAI(base_url=config.get("backend_url"))
            # Create collection without embedding function (we'll provide embeddings manually)
            self.situation_collection = self.chroma_client.create_collection(name=name)
        else:
            # Use ChromaDB's built-in default embeddings (no OpenAI needed!)
            # This uses sentence-transformers which runs locally
            self.situation_collection = self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

    def get_embedding(self, text):
        """Get embedding for a text"""
        if self.use_openai_embeddings:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        else:
            # ChromaDB will handle embeddings automatically - return None
            return None

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))

        add_params = {
            "documents": situations,
            "metadatas": [{"recommendation": rec} for rec in advice],
            "ids": ids,
        }
        
        # Only provide embeddings if using OpenAI (ChromaDB generates them automatically otherwise)
        if self.use_openai_embeddings:
            embeddings = [self.get_embedding(situation) for situation in situations]
            add_params["embeddings"] = embeddings
        
        self.situation_collection.add(**add_params)

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using embeddings"""
        
        query_params = {
            "n_results": n_matches,
            "include": ["metadatas", "documents", "distances"],
        }
        
        if self.use_openai_embeddings:
            # Provide query embedding for OpenAI
            query_embedding = self.get_embedding(current_situation)
            query_params["query_embeddings"] = [query_embedding]
        else:
            # ChromaDB will generate embedding from query text automatically
            query_params["query_texts"] = [current_situation]
        
        results = self.situation_collection.query(**query_params)

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
