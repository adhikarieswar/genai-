import random
import streamlit as st
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import time
import folium
from streamlit_folium import folium_static
import pandas as pd
import re
import json
from typing import Dict, List, Optional

st.set_page_config(page_title="Zomato Support", page_icon="🍔", layout="wide")

GROQ_API_KEY = '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'
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
st.markdown("""
<style>
    .tracking-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);   
    }
    .status-badge {
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    .preparing {
        background-color: #fff3cd;
        color: #856404;
    }
    .on-the-way {
        background-color: #cce5ff;
        color: #004085;
    }
    .nearby {
        background-color: #d4edda;
        color: #155724;
    }
    .delivered {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .order-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .issue-resolution {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 4px 4px 0;
    }
    .urgent-issue {
        background-color: #ffdddd;
        border-left: 6px solid #f44336;
        padding: 12px;
        margin: 10px 0;
    }
    .resolution-box {
        background-color: #ddffdd;
        border-left: 6px solid #4CAF50;
        padding: 12px;
        margin: 10px 0;
    }
    .history-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4e73df;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
required_session_vars = {
    "messages": [],
    "order_id": None,
    "tracking_active": False,
    "delivery_progress": 0,
    "rating_submitted": False,
    "show_history": False,
    "visible_history": [],
    "user_email": "adhikarieswar44@gmail.com",
    "current_issue": None,
    "personal_orders_loaded": False
}

for var, default in required_session_vars.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ==================== DATABASES ====================
# 1. PERSONAL USER DATABASE (Your Orders)
PERSONAL_ORDER_DATABASE = {
    "adhikarieswar44@gmail.com": {
        "user_profile": {
            "name": "Eswar Adhikari",
            "join_date": "2022-05-15",
            "loyalty_points": 1250,
            "preferred_addresses": [
                "12th Main Rd, Koramangala, Bangalore",
                "24th Main Rd, HSR Layout, Bangalore"
            ],
            "favorite_restaurants": ["Burger King", "McDonald's", "Domino's"]
        },
        "order_history": {
            "ZO3709": {
                "restaurant": "Burger King - Andheri East",
                "items": [
                    {"name": "Whopper Meal", "price": 199, "quantity": 1},
                    {"name": "Chicken Nuggets", "price": 99, "quantity": 2}
                ],
                "timestamps": {
                    "ordered": "2024-03-10 18:30:00",
                    "estimated_delivery": "2024-03-10 19:10:00",
                    "actual_delivery": "2024-03-10 19:22:00"
                },
                "delivery_details": {
                    "address": "12th Main Rd, Koramangala, Bangalore",
                    "partner": {
                        "name": "Rahul",
                        "phone": "919876543210",
                        "rating": 4.2
                    }
                },
                "payment": {
                    "method": "UPI",
                    "status": "completed",
                    "amount": 397
                },
                "issues": [
                    {
                        "type": "late_delivery",
                        "delay_minutes": 12,
                        "resolution": {
                            "status": "resolved",
                            "compensation": "50 Zomato points",
                            "date_resolved": "2024-03-10"
                        }
                    }
                ],
                "feedback": {
                    "food_rating": 4,
                    "delivery_rating": 3,
                    "comments": "Good but late"
                }
            },
            "ZO5636": {
                "restaurant": "McDonald's - Koramangala",
                "items": [
                    {"name": "McAloo Tikki Burger", "price": 49, "quantity": 2},
                    {"name": "McFlurry Oreo", "price": 79, "quantity": 1}
                ],
                "timestamps": {
                    "ordered": "2024-03-08 12:45:00",
                    "estimated_delivery": "2024-03-08 13:20:00",
                    "actual_delivery": "2024-03-08 13:25:00"
                },
                "delivery_details": {
                    "address": "12th Main Rd, Koramangala, Bangalore",
                    "partner": {
                        "name": "Vijay",
                        "phone": "919876543211",
                        "rating": 4.5
                    }
                },
                "payment": {
                    "method": "Credit Card",
                    "status": "completed",
                    "amount": 177
                },
                "issues": [
                    {
                        "type": "missing_item",
                        "description": "Missing 1 McFlurry Oreo",
                        "resolution": {
                            "status": "pending",
                            "compensation": "Refund of ₹79",
                            "date_reported": "2024-03-08"
                        }
                    }
                ],
                "feedback": {
                    "food_rating": 3,
                    "delivery_rating": 2,
                    "comments": "Missing item in order"
                }
            },
            "ZO7845": {
                "restaurant": "Domino's Pizza - Connaught Place",
                "items": [
                    {"name": "Farmhouse Pizza", "price": 299, "quantity": 1},
                    {"name": "Garlic Bread", "price": 99, "quantity": 1}
                ],
                "timestamps": {
                    "ordered": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "estimated_delivery": (datetime.now() + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
                },
                "delivery_details": {
                    "address": "12th Main Rd, Koramangala, Bangalore",
                    "partner": {
                        "name": "Amit",
                        "phone": "919876543212",
                        "rating": 4.7
                    }
                },
                "payment": {
                    "method": "Cash on Delivery",
                    "status": "pending",
                    "amount": 398
                },
                "issues": [],
                "feedback": None
            }
        }
    }
}


# ==================== CORE FUNCTIONS ====================
def get_personal_orders(email: str) -> Dict[str, Dict]:
    """Get all orders for a specific user from personal database"""
    return PERSONAL_ORDER_DATABASE.get(email, {}).get("order_history", {})


def initialize_user_orders():
    """Load user's personal orders into session"""
    if not st.session_state.personal_orders_loaded:
        personal_orders = get_personal_orders(st.session_state.user_email)
        for order_id in personal_orders.keys():
            if order_id not in st.session_state.visible_history:
                st.session_state.visible_history.append(order_id)
        st.session_state.personal_orders_loaded = True


def get_combined_order_data(order_id: str) -> Optional[Dict]:
    """Get order data from either personal or system database"""
    # Check personal orders first
    personal_order = get_personal_orders(st.session_state.user_email).get(order_id)
    if personal_order:
        return personal_order

    # Check system database
    orders_db = generate_system_orders()
    return orders_db.get(order_id)


def generate_system_orders():
    orders = {}
    # First load all personal orders
    personal_orders = get_personal_orders(st.session_state.user_email)
    for order_id, details in personal_orders.items():
        orders[order_id] = details

    # Generate additional demo orders
    restaurants = ["McDonald's", "KFC", "Pizza Hut", "Subway"]
    items_db = [
        {"name": "Cheeseburger", "price": 99},
        {"name": "Fries", "price": 59},
        {"name": "Pizza", "price": 199},
        {"name": "Pasta", "price": 179}
    ]

    for i in range(10):
        order_id = f"ZO{random.randint(1000, 9999)}"
        if order_id not in orders:
            selected_items = random.sample(items_db, random.randint(1, 3))
            total = sum(item["price"] for item in selected_items)

            orders[order_id] = {
                "restaurant": f"{random.choice(restaurants)} - {random.choice(['Koramangala', 'Andheri', 'Connaught Place'])}",
                "items": selected_items,
                "status": random.choice(["preparing", "on the way", "delivered"]),
                "order_time": (datetime.now() - timedelta(days=random.randint(0, 6))).strftime("%Y-%m-%d %H:%M:%S"),
                "delivery_address": random.choice(["12th Main Rd, Koramangala", "24th Main Rd, HSR Layout"]),
                "total_amount": total,
                "delivery_partner": {
                    "name": random.choice(['Rahul', 'Vijay', 'Amit', 'Priya']),
                    "phone": f"91{random.randint(7000000000, 9999999999)}",
                    "rating": round(random.uniform(3.5, 5.0), 1)
                },
                "payment_method": random.choice(["UPI", "Credit Card", "Cash on Delivery"]),
                "issues": [],
                "rating": None
            }
    return orders


def analyze_order_issues(order_id: str) -> Dict:
    """Comprehensive analysis of order issues with automatic resolutions"""
    order = get_combined_order_data(order_id)
    if not order:
        return {"error": "Order not found"}

    analysis = {
        "issues_found": [],
        "suggested_resolutions": [],
        "compensation_offered": []
    }

    # 1. Check for late delivery
    if "timestamps" in order and "actual_delivery" in order["timestamps"]:
        ordered_time = datetime.strptime(order["timestamps"]["ordered"], "%Y-%m-%d %H:%M:%S")
        actual_time = datetime.strptime(order["timestamps"]["actual_delivery"], "%Y-%m-%d %H:%M:%S")
        estimated_time = datetime.strptime(order["timestamps"]["estimated_delivery"], "%Y-%m-%d %H:%M:%S")

        if actual_time > estimated_time:
            delay_minutes = (actual_time - estimated_time).seconds // 60
            analysis["issues_found"].append(f"Late delivery ({delay_minutes} minutes)")
            if delay_minutes > 15:
                analysis["suggested_resolutions"].append("Full refund of delivery charges")
                analysis["compensation_offered"].append("100 Zomato points")
            else:
                analysis["suggested_resolutions"].append("50% discount on next order")
                analysis["compensation_offered"].append("50 Zomato points")

    # 2. Check for missing/wrong items
    if "issues" in order:
        for issue in order["issues"]:
            if "missing" in issue["type"].lower():
                analysis["issues_found"].append("Missing items in order")
                analysis["suggested_resolutions"].append("Full refund for missing items")
                analysis["compensation_offered"].append("20% discount coupon")
            elif "wrong" in issue["type"].lower():
                analysis["issues_found"].append("Wrong items delivered")
                analysis["suggested_resolutions"].append("Replacement order or full refund")
                analysis["compensation_offered"].append("Free dessert on next order")

    # 3. Check payment issues
    if "payment" in order and order["payment"]["status"] != "completed":
        analysis["issues_found"].append("Payment issue")
        analysis["suggested_resolutions"].append("Retry payment or change method")
        analysis["compensation_offered"].append("10% discount on successful payment")

    # Generate resolution message
    if analysis["issues_found"]:
        analysis["resolution_message"] = (
                "🚨 We've identified the following issues:\n\n" +
                "\n".join(f"• {issue}" for issue in analysis["issues_found"]) +
                "\n\n🛠️ Suggested Resolutions:\n" +
                "\n".join(f"• {res}" for res in analysis["suggested_resolutions"]) +
                "\n\n💰 Compensation Offered:\n" +
                "\n".join(f"• {comp}" for comp in analysis["compensation_offered"]) +
                "\n\nWe sincerely apologize for the inconvenience."
        )
    else:
        analysis["resolution_message"] = "No issues detected with this order."

    return analysis


def extract_order_id(text: str) -> Optional[str]:
    """Extract order ID from text using regex"""
    match = re.search(r'ZO\d{4}', text)
    return match.group(0) if match else None


def sync_order_history():
    """Ensure all historical orders are in the database"""
    personal_orders = get_personal_orders(st.session_state.user_email)
    for order_id in personal_orders:
        if order_id not in st.session_state.visible_history:
            st.session_state.visible_history.append(order_id)


def show_delivery_map(order: Dict):
    """Display the delivery tracking map"""
    if not order:
        return

    rest_loc = (12.9352, 77.6245)  # Default restaurant location
    cust_loc = (12.9279, 77.6271)  # Default customer location
    curr_loc = rest_loc  # Default current location

    m = folium.Map(location=[(rest_loc[0] + cust_loc[0]) / 2, (rest_loc[1] + cust_loc[1]) / 2], zoom_start=13)

    folium.Marker(
        rest_loc,
        popup=f"<b>{order.get('restaurant', 'Restaurant')}</b>",
        icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
    ).add_to(m)

    folium.Marker(
        cust_loc,
        popup=f"<b>Delivery Address</b><br>{order.get('delivery_address', '')}",
        icon=folium.Icon(color="blue", icon="home", prefix="fa")
    ).add_to(m)

    folium.Marker(
        curr_loc,
        popup=f"<b>Delivery Partner</b><br>{order.get('delivery_partner', {}).get('name', '')}",
        icon=folium.Icon(color="green", icon="motorcycle", prefix="fa")
    ).add_to(m)

    folium.PolyLine(
        locations=[rest_loc, curr_loc, cust_loc],
        color="#4e73df",
        weight=3,
        opacity=0.8
    ).add_to(m)

    folium_static(m, width=700, height=400)


def show_order_timeline(order: Dict):
    """Display the order progress timeline"""
    if not order:
        return

    status = order.get('status', 'preparing')
    progress = st.session_state.delivery_progress / 100

    timeline = [
        {"step": "Order Confirmed", "icon": "📦", "completed": True},
        {"step": "Preparing", "icon": "👨‍🍳", "completed": status != 'preparing'},
        {"step": "On the Way", "icon": "🛵", "completed": status in ['on the way', 'nearby', 'delivered']},
        {"step": "Nearby", "icon": "📍", "completed": status in ['nearby', 'delivered']},
        {"step": "Delivered", "icon": "✅", "completed": status == 'delivered'}
    ]

    for stage in timeline:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f"<div style='font-size: 24px; text-align: center;'>{stage['icon']}</div>",
                        unsafe_allow_html=True)
        with col2:
            if stage["completed"]:
                st.markdown(f"<div style='color: #28a745; font-weight: 500;'>{stage['step']} ✔️</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: #6c757d;'>{stage['step']}</div>", unsafe_allow_html=True)


def update_delivery_progress():
    """Update the delivery progress status"""
    if st.session_state.order_id:
        order = get_combined_order_data(st.session_state.order_id)
        if order and 'status' in order:
            if order['status'] == 'preparing':
                st.session_state.delivery_progress = 25
            elif order['status'] == 'on the way':
                st.session_state.delivery_progress = 60
            elif order['status'] == 'nearby':
                st.session_state.delivery_progress = 85
            elif order['status'] == 'delivered':
                st.session_state.delivery_progress = 100


# ==================== TOOL DEFINITIONS ====================
@tool
def search_zomato_help_center(query: str) -> str:
    """Search Zomato's help center documentation for answers to common questions."""
    try:
        loader = WebBaseLoader(["https://www.zomato.com/help"])
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Could not access help center: {str(e)}"


@tool
def get_order_details(order_id: str) -> dict:
    """Retrieve detailed information about a specific order from either personal or system database."""
    sync_order_history()
    order_data = get_combined_order_data(order_id)
    if order_data:
        return order_data
    return {"error": f"Order {order_id} not found in our system"}


@tool
def get_order_issues(order_id: str) -> dict:
    """Get detailed analysis of any issues with an order and suggested resolutions."""
    return analyze_order_issues(order_id)


@tool
def get_user_profile(email: str) -> dict:
    """Retrieve the user's profile information including order preferences and history."""
    return PERSONAL_ORDER_DATABASE.get(email, {}).get("user_profile", {})


@tool
def get_faq_information() -> str:
    """Retrieve frequently asked questions about Zomato orders and services."""
    faqs = [
        {"question": "How can I track my order?",
         "answer": "You can track your order in real-time using the order tracking feature in the app or website."},
        {"question": "What payment methods are accepted?",
         "answer": "We accept credit/debit cards, net banking, UPI, and cash on delivery."},
        {"question": "How do I cancel an order?",
         "answer": "You can cancel your order from the order details page if the restaurant hasn't started preparing it yet."},
        {"question": "What is Zomato's refund policy?",
         "answer": "Refunds are processed within 5-7 business days depending on your payment method."},
        {"question": "How do I report an issue with my order?",
         "answer": "You can report issues directly through the order details page or by chatting with our support team."}
    ]
    return "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in faqs])


tools = [search_zomato_help_center, get_order_details, get_order_issues, get_user_profile, get_faq_information]


# ==================== AGENT SETUP ====================
def get_llm():
    return ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=GROQ_API_KEY,
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.5,
        max_tokens=1000,
        streaming=True
    )


def get_agent():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are Zomato's AI support assistant. Your responsibilities include:
        1. Providing accurate order information
        2. Resolving issues with intelligent analysis
        3. Answering questions about Zomato services
        4. Offering personalized recommendations

        SPECIAL INSTRUCTIONS:
        - ALWAYS check for order IDs in messages
        - For order-specific questions, use get_order_details
        - For issues, use get_order_issues
        - For user preferences, use get_user_profile
        - Be proactive in suggesting solutions
        - Never make up information
        - If unsure, ask clarifying questions

        USER PROFILE:
        Name: {user_name}
        Email: {user_email}
        Loyalty Points: {loyalty_points}
        """.format(
            user_name=PERSONAL_ORDER_DATABASE[st.session_state.user_email]["user_profile"]["name"],
            user_email=st.session_state.user_email,
            loyalty_points=PERSONAL_ORDER_DATABASE[st.session_state.user_email]["user_profile"]["loyalty_points"]
        )),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def process_agent_response(response: str, order_id: str = None) -> str:
    """Post-process agent responses with order context and formatting."""
    if order_id:
        if order_id not in response:
            response = f"Regarding order {order_id}:\n\n{response}"

    # Add formatting for issues
    if "issue" in response.lower() or "problem" in response.lower():
        response = f"<div class='issue-resolution'>{response}</div>"

    return response


def handle_agent_error(error: Exception, order_id: str = None) -> str:
    """Handle agent errors gracefully with order context."""
    error_msg = str(error)
    if "Order" in error_msg and order_id:
        return f"Sorry, I encountered an issue retrieving information for order {order_id}. Please try again later."
    return "Sorry, I'm having trouble processing your request. Please try again."


# ==================== MAIN APP ====================
def main():
    st.title("🍔 Zomato Order Support")
    st.caption(
        f"Personalized support for {PERSONAL_ORDER_DATABASE[st.session_state.user_email]['user_profile']['name']}")

    # Initialize user data
    initialize_user_orders()

    # Display user profile in sidebar
    with st.sidebar:
        user_profile = PERSONAL_ORDER_DATABASE[st.session_state.user_email]["user_profile"]
        st.markdown(f"""
            <div class='tracking-card'>
                <h3>👤 {user_profile['name']}</h3>
                <p>📧 {st.session_state.user_email}</p>
                <p>⭐ {user_profile['loyalty_points']} Loyalty Points</p>
                <p>📍 {user_profile['preferred_addresses'][0]}</p>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.order_id:
            order = get_combined_order_data(st.session_state.order_id)
            if order:
                status_class = order.get('status', '').replace(' ', '-')
                st.markdown(f"""
                    <div class='tracking-card'>
                        <h3>Order #{st.session_state.order_id}</h3>
                        <p><strong>Restaurant:</strong> {order.get('restaurant', '')}</p>
                        <p><strong>Status:</strong> 
                            <span class='status-badge {status_class}'>
                                {order.get('status', '').title()}
                            </span>
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(st.session_state.delivery_progress / 100)

                if st.button("📞 Contact Delivery Partner"):
                    partner = order.get('delivery_partner', {})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Connecting you with {partner.get('name', 'delivery partner')} at {partner.get('phone', '')}"
                    })

                if st.button("🔄 Refresh Status"):
                    update_delivery_progress()
                    st.rerun()

        st.markdown("---")
        st.subheader("Quick Actions")
        if st.button("📋 Order History"):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

        if st.button("❗ Report Issue"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Please describe the issue you're facing with your order."
            })

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Support", "📦 Order Tracking", "📜 Order History"])

    with tab1:
        # Chat container
        chat_container = st.container(height=500)

        # Display chat messages
        for message in st.session_state.messages:
            with chat_container:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("How can I help you today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Extract order ID
            order_id = extract_order_id(prompt)
            if order_id:
                st.session_state.order_id = order_id
                st.session_state.tracking_active = True

            # Generate response
            try:
                agent = get_agent()
                response = agent.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages
                })
                response_text = response["output"]
            except Exception as e:
                response_text = handle_agent_error(e, order_id)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

        # Show issue analysis if available
        if st.session_state.order_id:
            issue_analysis = analyze_order_issues(st.session_state.order_id)
            if issue_analysis.get("issues_found"):
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(f"""
                            <div class='urgent-issue'>
                                <h4>⚠️ Order Issues Detected</h4>
                                {issue_analysis["resolution_message"]}
                            </div>
                        """, unsafe_allow_html=True)

    with tab2:
        if st.session_state.order_id:
            order = get_combined_order_data(st.session_state.order_id)
            if order:
                st.subheader("📍 Live Order Tracking")
                show_delivery_map(order)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🚚 Delivery Progress")
                    show_order_timeline(order)

                    if order.get('status') != 'delivered':
                        eta = (100 - st.session_state.delivery_progress) / 100 * 30  # 30 min estimate
                        st.metric("Estimated Time Remaining", f"{int(eta)} minutes")

                with col2:
                    st.subheader("🍽️ Order Details")
                    st.markdown(f"""
                        <div class='order-card'>
                            <p><strong>Restaurant:</strong> {order.get('restaurant', '')}</p>
                            <p><strong>Order Time:</strong> {order.get('order_time', '')}</p>
                            <p><strong>Delivery Address:</strong> {order.get('delivery_address', '')}</p>
                            <p><strong>Total Amount:</strong> ₹{order.get('total_amount', 0)}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    with st.expander("View Items"):
                        for item in order.get("items", []):
                            st.write(
                                f"- {item.get('name', 'Item')} (₹{item.get('price', 0)} x {item.get('quantity', 1)})")

                    if order.get('status') == 'delivered' and not st.session_state.rating_submitted:
                        st.subheader("🌟 Rate Your Experience")
                        rating = st.slider("Rating", 1, 5, 4)
                        feedback = st.text_area("Comments (optional)")

                        if st.button("Submit Rating"):
                            st.session_state.rating_submitted = True
                            st.success("Thank you for your feedback!")
                            st.balloons()
        else:
            st.info("Enter your order ID in the chat to track your delivery")

    with tab3:
        st.subheader("📜 Your Order History")

        # Get all orders sorted by date (newest first)
        all_orders = get_personal_orders(st.session_state.user_email)
        sorted_orders = sorted(
            all_orders.items(),
            key=lambda x: x[1].get("timestamps", {}).get("ordered", ""),
            reverse=True
        )

        if not sorted_orders:
            st.info("You haven't placed any orders yet.")
        else:
            for order_id, order in sorted_orders:
                with st.container():
                    st.markdown(f"""
                        <div class='history-card'>
                            <h4>Order #{order_id} - {order.get('restaurant', '')}</h4>
                            <p><strong>Date:</strong> {order.get('timestamps', {}).get('ordered', '')}</p>
                            <p><strong>Status:</strong> {order.get('status', 'delivered').title()}</p>
                            <p><strong>Total:</strong> ₹{order.get('total_amount', order.get('payment', {}).get('amount', 0))}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"View details for Order #{order_id}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Order Details")
                            st.write(f"**Restaurant:** {order.get('restaurant', '')}")
                            st.write(f"**Order Time:** {order.get('timestamps', {}).get('ordered', '')}")
                            if 'actual_delivery' in order.get('timestamps', {}):
                                st.write(f"**Delivered At:** {order['timestamps']['actual_delivery']}")
                            st.write(f"**Delivery Address:** {order.get('delivery_address', '')}")
                            st.write(
                                f"**Payment Method:** {order.get('payment_method', order.get('payment', {}).get('method', ''))}")
                            st.write(
                                f"**Total Amount:** ₹{order.get('total_amount', order.get('payment', {}).get('amount', 0))}")

                        with col2:
                            st.subheader("Items Ordered")
                            for item in order.get("items", []):
                                st.write(
                                    f"- {item.get('name', 'Item')} (₹{item.get('price', 0)} x {item.get('quantity', 1)})")

                            if order.get('feedback'):
                                st.subheader("Your Feedback")
                                st.write(f"Food Rating: {order['feedback']['food_rating']}/5")
                                st.write(f"Delivery Rating: {order['feedback']['delivery_rating']}/5")
                                if order['feedback'].get('comments'):
                                    st.write(f"Comments: {order['feedback']['comments']}")

                        if order.get('issues'):
                            st.subheader("Reported Issues")
                            for issue in order['issues']:
                                st.markdown(f"""
                                    <div class='issue-resolution'>
                                        <p><strong>Type:</strong> {issue.get('type', '').replace('_', ' ').title()}</p>
                                        {f"<p><strong>Description:</strong> {issue.get('description', '')}</p>" if issue.get('description') else ""}
                                        {f"<p><strong>Resolution:</strong> {issue.get('resolution', {}).get('compensation', '')}</p>" if issue.get('resolution', {}).get('compensation') else ""}
                                        <p><strong>Status:</strong> {issue.get('resolution', {}).get('status', '').title()}</p>
                                    </div>
                                """, unsafe_allow_html=True)

    # Auto-update for active orders
    if st.session_state.tracking_active and st.session_state.order_id:
        time.sleep(1)
        update_delivery_progress()
        st.rerun()


if __name__ == "__main__":
    main()