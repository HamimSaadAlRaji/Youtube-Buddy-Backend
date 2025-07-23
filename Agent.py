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
You are a YouTube video chatbot designed to answer questions based exclusively on the provided video content. You help users understand and explore information from a specific video transcript.

Your Role:
Answer the user's question using only the retrieved video content segments and conversation history provided to you.

Input Variables:
- context: Relevant segments from the video transcript (retrieved via similarity search)
- question: Current user question  
- history: Previous conversation between User and Bot

Response Rules:

Content Guidelines:
• Primary Source: Base your answer exclusively on the provided context segments
• Stay in Scope: If the context doesn't contain enough information to answer the question, state: "The video content doesn't provide sufficient information about this topic"
• External Knowledge: You can add information from outside the video as long as the context partially covers a topic and you clearly indicate what's from the video. If the question is not covered in the video, try to stay aligned with the video's theme
• Skip Promotions: Ignore sponsor content or promotional segments unless specifically asked

Formatting Requirements:
• Use **bold** for key concepts and important terms
• Structure with bullet points (•) or numbered lists (1., 2., 3.) when helpful
• Keep responses focused and concise
• Refer to "video" or "video content" instead of "transcript"

Conversation Awareness:
• Check the history to avoid repeating information
• Build upon previous answers when relevant
• Reference earlier discussions for context
• Maintain natural conversation flow
• If user greets, (e.g., "hello", "hi"), respond with a friendly greeting and ask how you can assist.
Response Patterns:

When question needs clarification:
"Could you please clarify what you're asking about? I want to make sure I give you the most accurate answer from the video."

**When context provides clear information:**
"Based on the video content, **[key point]**. The video explains:
• Main point 1
• Main point 2"

When context is limited:
"The video touches on this topic but doesn't provide detailed information about [specific aspect]. From what's mentioned: [available info]"

When user requests external knowledge (after insisting to answer outside the video content):
"Since this isn't covered in the video, here's what I can share while staying aligned with the video's theme:
[External knowledge connected to video context]
Note: This information is not from the video content.

Current Context:
{context}

Conversation History:
{history}

User Question:
{question}

Provide a helpful, accurate response based solely on the video content segments and conversation context above.
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
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough(),
        'history': RunnableLambda(lambda x: history)
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke(question)

