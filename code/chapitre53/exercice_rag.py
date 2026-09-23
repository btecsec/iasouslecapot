import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

class RAGAssistant:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Initialisation des composants de base
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 100, "temperature": 0.01}
        )
        self.vector_store = None
        self.qa_chain = None

    def index_document(self, file_path: str, chunk_size=500, chunk_overlap=50):
        """Lit un document, le découpe et l'indexe dans la base vectorielle."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        print(f"Document découpé en {len(chunks)} morceaux.")

        self.vector_store = Chroma.from_texts(chunks, self.embeddings)

        template = """<|user|>
SÉCURITÉ: Réponds UNIQUEMENT avec le contexte fourni.
Si la réponse n'est pas dans le contexte, réponds exactement "Je ne sais pas".
CONTEXTE :
{context}
---
QUESTION :
{question}

Réponse :
<|assistant|>"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str, threshold=1.6):
        """Pose une question avec garde-fou de distance vectorielle."""
        # 1. VÉRIFICATION DE DISTANCE (Guardrail)
        # Chroma utilise la distance L2 : plus c'est haut, moins c'est similaire.
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=1)

        if not docs_and_scores:
            return {"answer": "Je ne sais pas", "sources": []}

        score = docs_and_scores[0][1]
        print(f"DEBUG: Distance Score = {score:.4f} (Threshold: {threshold})")
        print(f"INFO: Distance L2 entre la question et le document. Plus le score est bas, plus le sens est proche.")

        if score > threshold:
            print(f"DEBUG: Blocage Guardrail -> Le score {score:.4f} est supérieur au seuil {threshold}, le contexte est jugé hors-sujet.")
            return {"answer": "Je ne sais pas", "sources": []}
        else:
            print(f"DEBUG: Validation Guardrail -> Le score {score:.4f} est inférieur au seuil {threshold}, tout est bon, appel du LLM.")

        # 2. APPEL LLM (seulement si pertinent)
        result = self.qa_chain.invoke({"query": question})
        answer = result["result"]

        if "<|assistant|>" in answer:
            answer = answer.split("<|assistant|>")[-1]

        for stop_word in ["<|system|>", "<|user|>", "Contexte :", "Question :"]:
            if stop_word in answer:
                answer = answer.split(stop_word)[0]

        return {
            "answer": answer.strip(),
            "sources": [doc.page_content for doc in result["source_documents"]]
        }

def main():
    doc_name = "mon_document.txt"
    content = """
    Le manuel de l'utilisateur du RobotX1000.
    Pour réinitialiser le mot de passe, maintenez le bouton 'Reset' pendant 5 secondes.
    Le RobotX1000 fonctionne avec une batterie Lithium-Ion de 5000mAh.
    Le support technique est joignable à l'adresse support@robotx.com.
    Le RobotX1000 ne peut pas voler, il se déplace uniquement au sol.
    """
    with open(doc_name, "w", encoding="utf-8") as f:
        f.write(content)

    assistant = RAGAssistant()
    assistant.index_document(doc_name)

    questions = [
        "Comment réinitialiser le mot de passe ?",
        "Quelle est la capacité de la batterie ?",
        "Où contacter le support ?",
        "Quelle est la météo à Paris ?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}: {q}")
        res = assistant.ask(q)
        print(f"Réponse : {res['answer']}")
        print("Sources utilisées :")
        for idx, src in enumerate(res['sources'], 1):
            print(f"  [{idx}] {src[:100]}...")

if __name__ == "__main__":
    main()
