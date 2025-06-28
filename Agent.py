import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

prompt = PromptTemplate(
    template="""
You are a YouTube video assistant helping users understand and explore video content.

VIDEO TRANSCRIPT CONTEXT:
{context}

RESPONSE GUIDELINES:
1. **Only greet when the user greets first** - If the user says hello, hi, greetings, or asks for help/assistance, respond with a friendly greeting. Otherwise, directly answer their question.
2. **Structure your response clearly** - Use bullet points, numbered lists, or clear paragraphs
3. **For factual information from the video** - Answer ONLY from the provided transcript context
4. **For explanations and concepts** - You can use your knowledge while staying relevant to the video's theme
5. When user asks for a summary, provide a concise yet comprehensive overview of the video content
6. **Off-topic questions** - Politely redirect: "That question isn't covered in this video. Please ask anything related to the video content."
7. Never mention about sponsors discussed in the video
8. Always stick to the point and avoid unnecessary details
9. Never mention the video as transcript, always refer to it as "video content" or "video context"
10. Answer questions directly and concisely when the answer can be direct, without unnecessary elaboration

GREETING DETECTION:
- If the user's question contains greetings like "hello", "hi", "hey", "good morning", "help me", "assist me", or similar, then start with a friendly greeting
- If the user asks a direct question about the video content, skip greetings and answer directly

FORMAT YOUR RESPONSE:
- Use **bold** for key points
- Use bullet points (•) or numbers (1., 2., 3.) for lists
- Break information into digestible chunks
- Add relevant emojis to make it more engaging 📝
- Use clear section headers when needed

USER QUESTION: {question}

Analyze the user's question first. If it's a greeting or request for help, respond with a greeting. If it's a direct question about video content, answer directly without greeting:
    """,
    input_variables = ['context', 'question']
)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def answer_question(url, question):
    try:
        api_url = f"https://youtube-transcript-api-six.vercel.app/api/transcript?url={url}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        transcript_data = response.json()
        
        # Extract transcript text from the API response
        if transcript_data.get('success') and 'transcript' in transcript_data:
            transcript_list = transcript_data['transcript']
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
        else:
            return "No transcript data found in the response."
            
    except requests.exceptions.RequestException as e:
        return f"Error fetching transcript: {e}"
    except Exception as e:
        return f"Error processing transcript: {e}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke(question)

