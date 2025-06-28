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
1. **Structure your response clearly** - Use bullet points, numbered lists, or clear paragraphs
2. **For factual information from the video** - Answer ONLY from the provided transcript context
3. **For explanations and concepts** - You can use your knowledge while staying relevant to the video's theme
4. **For summaries** - Include ALL important points, don't miss key details
5. **Make responses engaging** - Use clear headings, emojis where appropriate, and conversational tone
6. **Off-topic questions** - Politely redirect: "That question isn't covered in this video. Please ask anything related to the video content."

FORMAT YOUR RESPONSE:
- Use **bold** for key points
- Use bullet points (•) or numbers (1., 2., 3.) for lists
- Break information into digestible chunks
- Add relevant emojis to make it more engaging 📝
- Use clear section headers when needed

USER QUESTION: {question}

Provide a well-structured, engaging, and comprehensive answer:
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

