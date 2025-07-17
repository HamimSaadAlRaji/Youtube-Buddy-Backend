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
You are a YouTube video content assistant that helps users understand video information.

VIDEO CONTENT:
{context}

CONVERSATION HISTORY:
{history}

CORE RULES:
1. **STRICT CONTENT BOUNDARY**: Only answer questions about topics present in the video content above
2. **SUPPLEMENT WHEN NEEDED**: If a topic is mentioned but lacks detail, enhance with relevant knowledge while staying aligned with the video's theme
3. **REJECT OFF-TOPIC**: If the question is completely unrelated to the video content, respond: "This topic isn't covered in the video. Please ask about something from the video content."
4. **IGNORE GREETINGS**: Skip "hi", "hello", "hey" - jump straight to answering the question
5. **STRUCTURED RESPONSES**: Always format clearly with bullet points or numbered lists
6. **CONVERSATION AWARENESS**: Use the conversation history to provide contextual responses and avoid repeating information

RESPONSE FORMAT:
• Use **bold** for key concepts and important terms
• Keep answers focused and concise - highlight main points only
• Structure with bullet points (•) or numbers (1., 2., 3.)
• Avoid mentioning "transcript" - refer to "video content" instead
• Skip sponsor mentions or unrelated promotional content

USER QUESTION: {question}

PROCESS:
1. Check if the question relates to any topic in the video content
2. Consider the conversation history for context
3. If YES: Answer using video information, supplement with knowledge if the video lacks detail
4. If NO: Politely decline and redirect to video-related topics
    """,
    input_variables=['context', 'question', 'history']
)

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def answer_question(context):
    try:
        # Extract transcript from the context object
        transcript_data = context['context']['transcript']
        transcript = " ".join(chunk["text"] for chunk in transcript_data)
        
        # Extract conversation history from messages
        messages = context['context']['messages']
        print("Messages:", messages)
        history = ""
        question = ""
        
        # Process messages to build history and get the last question
        for i, message in enumerate(messages):
            if message['sender'] == 'user':
                print(message['sender'])
                question = message['message']
                print("Question:", question)
            
            # Build conversation history
            sender = "User" if message['sender'] == 'user' else "Bot"
            history += f"{sender}: {message['message']}\n"
        
        if not transcript:
            return "No transcript data found in the context."
        
        if not question:
            return "No question found in the conversation history."
            
    except Exception as e:
        return f"Error processing context: {e}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough(),
        'history': RunnableLambda(lambda x: history)
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke(question)

