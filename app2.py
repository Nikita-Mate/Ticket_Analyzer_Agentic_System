import streamlit as st
import asyncio
import json
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import os
st.set_page_config(
    page_title="Customer Support Ticket Analyzer",
    page_icon="🎫",
    layout="wide"
)
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .analysis-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .metric-item {
        text-align: center;
        padding: 0.5rem;
    }
    .priority-high { color: #f44336; font-weight: bold; }
    .priority-medium { color: #ff9800; font-weight: bold; }
    .priority-low { color: #4caf50; font-weight: bold; }
    .priority-critical { color: #d32f2f; font-weight: bold; background-color: #ffebee; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "API Key has been removed for security purpose"
class CustomerTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
class Department(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"
    ESCALATION = "escalation"
class TicketInput(BaseModel):
    ticket_id: str
    customer_tier: CustomerTier
    subject: str
    message: str
    previous_tickets: int
    monthly_revenue: float
    account_age_days: int
class TechnicalAnalysis(BaseModel):
    complexity_score: int = Field(ge=1, le=10, description="Technical complexity 1-10")
    category: str = Field(description="Technical category")
    requires_specialist: bool = Field(description="Needs technical specialist")
    estimated_resolution_hours: int = Field(description="Estimated hours to resolve")
    keywords: List[str] = Field(description="Key technical terms identified")
class BusinessAnalysis(BaseModel):
    customer_value_score: int = Field(ge=1, le=10, description="Customer value 1-10")
    churn_risk: int = Field(ge=1, le=10, description="Risk of customer leaving 1-10")
    revenue_impact: str = Field(description="Potential revenue impact")
    relationship_health: str = Field(description="Customer relationship status")
    escalation_recommended: bool = Field(description="Should this be escalated")
class RoutingDecision(BaseModel):
    department: Department
    priority: Priority
    assigned_team: str
    sla_hours: int = Field(description="Response time in hours")
    special_instructions: str
    confidence_score: float = Field(ge=0.0, le=1.0)
class ConflictResolution(BaseModel):
    final_decision: RoutingDecision
    reasoning: str
    technical_weight: float = Field(ge=0.0, le=1.0)
    business_weight: float = Field(ge=0.0, le=1.0)
    compromise_made: bool
@dataclass
class TechnicalContext:
    ticket: TicketInput
    analysis: Optional[TechnicalAnalysis] = None
@dataclass
class BusinessContext:
    ticket: TicketInput
    analysis: Optional[BusinessAnalysis] = None
@dataclass
class RoutingContext:
    ticket: TicketInput
    technical_analysis: Optional[TechnicalAnalysis] = None
    business_analysis: Optional[BusinessAnalysis] = None
    technical_routing: Optional[RoutingDecision] = None
    business_routing: Optional[RoutingDecision] = None
@st.cache_resource
def initialize_agents():
    technical_agent = Agent(
        'openai:gpt-4o-mini',
        result_type=TechnicalAnalysis,
        system_prompt="""You are a Senior Technical Support Analyst with 10+ years experience.
        Analyze support tickets for technical complexity, categorization, and resource requirements.
        Focus on:
        - Technical complexity and difficulty
        - Required expertise level
        - Time estimates based on technical scope
        - Technical keywords and patterns
        Be precise and technical in your analysis."""
    )
    business_agent = Agent(
        'openai:gpt-4o-mini',
        result_type=BusinessAnalysis,
        system_prompt="""You are a Customer Success Manager focused on business impact and relationships.
        Analyze tickets from customer value, retention, and business perspective.
        Focus on:
        - Customer lifetime value and tier importance
        - Churn risk assessment
        - Revenue impact potential
        - Relationship health indicators
        Prioritize high-value customers and retention risks."""
    )
    routing_agent = Agent(
        'openai:gpt-4o-mini',
        result_type=RoutingDecision,
        system_prompt="""You are a Support Operations Manager responsible for optimal ticket routing.
        Make routing decisions based on technical complexity and business impact.
        Routing Guidelines:
        - Enterprise customers get faster SLA
        - Complex technical issues go to specialists
        - High churn risk customers get priority
        - Balance workload across teams
        Provide clear routing with confidence scores."""
    )
    conflict_resolver = Agent(
        'openai:gpt-4o-mini',
        result_type=ConflictResolution,
        system_prompt="""You are the Support Director who resolves conflicts between technical and business priorities.
        When agents disagree on routing, make the final balanced decision.
        Consider:
        - Customer impact vs technical complexity
        - Resource availability
        - SLA commitments
        - Overall support strategy
        Explain your reasoning clearly."""
    )
    return technical_agent, business_agent, routing_agent, conflict_resolver
def append_chat_to_file(ticket_data: Dict[str, Any], result: Dict[str, Any]):
    chat_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticket_id": ticket_data.get("ticket_id", "N/A"),
        "subject": ticket_data.get("subject", ""),
        "customer_tier": ticket_data.get("customer_tier", ""),
        "message": ticket_data.get("message", ""),
        "technical_analysis": result.get("technical_analysis", {}),
        "business_analysis": result.get("business_analysis", {}),
        "final_routing": result.get("final_routing", {}),
        "confidence": result.get("confidence", 0),
        "conflict_occurred": result.get("conflict_occurred", False)
    }
    try:
        with open("chathistory.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(chat_entry, indent=2))
            f.write("\n-----------------------------\n")  
    except Exception as e:
        st.error(f"Failed to write chat history: {e}")

class TicketAnalyzer:
    def __init__(self, agents):
        self.technical_agent, self.business_agent, self.routing_agent, self.conflict_resolver = agents
        self.analysis_history = []
    async def analyze_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis pipeline"""
        try:
            ticket = TicketInput(**ticket_data)
            # Step 1: Parallel technical and business analysis
            technical_task = self._technical_analysis(ticket)
            business_task = self._business_analysis(ticket)
            technical_analysis, business_analysis = await asyncio.gather(
                technical_task, business_task
            )            
            # Step 2: Independent routing recommendations
            routing_context = RoutingContext(
                ticket=ticket,
                technical_analysis=technical_analysis,
                business_analysis=business_analysis
            )
            technical_routing = await self._get_technical_routing(routing_context)
            business_routing = await self._get_business_routing(routing_context)
            final_routing = await self._resolve_routing_conflict(
                routing_context, technical_routing, business_routing
            )
            analysis_result = {
                "ticket_id": ticket.ticket_id,
                "timestamp": datetime.now().isoformat(),
                "ticket_data": ticket_data,
                "technical_analysis": technical_analysis.model_dump(),
                "business_analysis": business_analysis.model_dump(),
                "final_routing": final_routing.model_dump(),
                "confidence": final_routing.confidence_score,
                "conflict_occurred": (technical_routing.priority != business_routing.priority or 
                                    technical_routing.department != business_routing.department)
            }
            self.analysis_history.append(analysis_result)
            return analysis_result       
        except Exception as e:
            return {"error": str(e)}
    async def _technical_analysis(self, ticket: TicketInput) -> TechnicalAnalysis:
        context = TechnicalContext(ticket=ticket)
        prompt = f"""
        Analyze this support ticket for technical complexity:
        Subject: {ticket.subject}
        Message: {ticket.message}
        Previous tickets: {ticket.previous_tickets}
        Provide detailed technical analysis including complexity score, category, 
        specialist requirements, time estimates, and key technical keywords.
        """
        result = await self.technical_agent.run(prompt, message_history=[])
        return result.data
    async def _business_analysis(self, ticket: TicketInput) -> BusinessAnalysis:
        context = BusinessContext(ticket=ticket)
        prompt = f"""
        Analyze this support ticket for business impact:
        Customer Tier: {ticket.customer_tier.value}
        Monthly Revenue: ${ticket.monthly_revenue}
        Account Age: {ticket.account_age_days} days
        Previous Tickets: {ticket.previous_tickets}
        Subject: {ticket.subject}
        Assess customer value, churn risk, revenue impact, and relationship health.
        """
        result = await self.business_agent.run(prompt, message_history=[])
        return result.data
    async def _get_technical_routing(self, context: RoutingContext) -> RoutingDecision:
        prompt = f"""
        Route this ticket based on TECHNICAL requirements:
        Technical Analysis:
        - Complexity: {context.technical_analysis.complexity_score}/10
        - Category: {context.technical_analysis.category}
        - Specialist needed: {context.technical_analysis.requires_specialist}
        - Est. hours: {context.technical_analysis.estimated_resolution_hours}
        Customer: {context.ticket.customer_tier.value}
        Subject: {context.ticket.subject}
        Focus on technical complexity and expertise requirements.
        """
        result = await self.routing_agent.run(prompt, message_history=[])
        return result.data
    async def _get_business_routing(self, context: RoutingContext) -> RoutingDecision:
        prompt = f"""
        Route this ticket based on BUSINESS priorities:
        Business Analysis:
        - Customer Value: {context.business_analysis.customer_value_score}/10
        - Churn Risk: {context.business_analysis.churn_risk}/10
        - Revenue Impact: {context.business_analysis.revenue_impact}
        - Escalation needed: {context.business_analysis.escalation_recommended}
        Customer: {context.ticket.customer_tier.value} (${context.ticket.monthly_revenue}/month)
        Focus on customer retention and business impact.
        """
        result = await self.routing_agent.run(prompt, message_history=[])
        return result.data
    async def _resolve_routing_conflict(
        self, 
        context: RoutingContext, 
        technical_routing: RoutingDecision, 
        business_routing: RoutingDecision
    ) -> RoutingDecision:
        if (technical_routing.priority != business_routing.priority or 
            technical_routing.department != business_routing.department or
            abs(technical_routing.sla_hours - business_routing.sla_hours) > 4):            
            prompt = f"""
            Resolve this routing conflict:
            TECHNICAL RECOMMENDATION:
            - Department: {technical_routing.department.value}
            - Priority: {technical_routing.priority.value}
            - SLA: {technical_routing.sla_hours}h
            - Team: {technical_routing.assigned_team}
            - Reasoning: {technical_routing.special_instructions}
            BUSINESS RECOMMENDATION:
            - Department: {business_routing.department.value}
            - Priority: {business_routing.priority.value}
            - SLA: {business_routing.sla_hours}h
            - Team: {business_routing.assigned_team}
            - Reasoning: {business_routing.special_instructions}
            Customer Context:
            - Tier: {context.ticket.customer_tier.value}
            - Revenue: ${context.ticket.monthly_revenue}/month
            - Technical Complexity: {context.technical_analysis.complexity_score}/10
            - Churn Risk: {context.business_analysis.churn_risk}/10
            Make the optimal balanced decision.
            """
            resolution = await self.conflict_resolver.run(prompt, message_history=[])
            return resolution.data.final_decision
        return business_routing
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analyzer' not in st.session_state:
    agents = initialize_agents()
    st.session_state.analyzer = TicketAnalyzer(agents)
def get_priority_class(priority):
    return f"priority-{priority.lower()}"
def display_analysis_result(result):
    if "error" in result:
        st.error(f"Analysis failed: {result['error']}")
        return
    ticket_data = result['ticket_data']
    technical = result['technical_analysis']
    business = result['business_analysis']
    routing = result['final_routing']
    st.markdown(f"""
    <div class="analysis-card">
        <h3>🎫 Ticket Analysis: {result['ticket_id']}</h3>
        <p><strong>Subject:</strong> {ticket_data['subject']}</p>
        <p><strong>Customer:</strong> {ticket_data['customer_tier'].title()} Tier</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Technical Complexity", f"{technical['complexity_score']}/10")
    with col2:
        st.metric("Customer Value", f"{business['customer_value_score']}/10")
    with col3:
        st.metric("Churn Risk", f"{business['churn_risk']}/10")
    with col4:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    priority_class = get_priority_class(routing['priority'])
    st.markdown(f"""
    <div class="analysis-card">
        <h4>✅ Final Routing Decision</h4>
        <p><strong>Department:</strong> {routing['department'].title()}</p>
        <p><strong>Priority:</strong> <span class="{priority_class}">{routing['priority'].upper()}</span></p>
        <p><strong>Assigned Team:</strong> {routing['assigned_team']}</p>
        <p><strong>SLA:</strong> {routing['sla_hours']} hours</p>
        <p><strong>Instructions:</strong> {routing['special_instructions']}</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("🔧 Technical Analysis Details"):
        st.write(f"Category: {technical['category']}")
        st.write(f"Requires Specialist: {'Yes' if technical['requires_specialist'] else 'No'}")
        st.write(f"Estimated Resolution:** {technical['estimated_resolution_hours']} hours")
        st.write(f"Keywords: {', '.join(technical['keywords'])}")
    with st.expander("💼 Business Analysis Details"):
        st.write(f"Revenue Impact: {business['revenue_impact']}")
        st.write(f"Relationship Health: {business['relationship_health']}")
        st.write(f"Escalation Recommended: {'Yes' if business['escalation_recommended'] else 'No'}")
    if result.get('conflict_occurred'):
        st.warning("⚠️ Routing conflict detected and resolved by conflict resolution agent")
def display_chat_message(message, is_user=True):
    css_class = "user-message" if is_user else "bot-message"
    icon = "👤" if is_user else "🤖"
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="margin-right: 0.5rem;">{icon}</span>
            <strong>{"You" if is_user else "Support AI"}</strong>
        </div>
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)
def main():
    st.title("🎫 Support Ticket Analyzer")
    st.markdown("**Multi-Agent AI System for Intelligent Ticket Routing**")
    with st.sidebar:
        st.header("📋 Quick Samples")
        if st.button("🔧 Technical Issue"):
            sample_data = {
                "ticket_id": "TECH-001",
                "customer_tier": "enterprise",
                "subject": "API returning 500 errors intermittently",
                "message": "Our production system has been failing with 500 errors for the past 2 hours. This is affecting customer transactions.",
                "previous_tickets": 2,
                "monthly_revenue": 25000,
                "account_age_days": 180
            }
            st.session_state.pending_analysis = sample_data
            st.rerun()
        if st.button("💳 Billing Issue"):
            sample_data = {
                "ticket_id": "BILL-002",
                "customer_tier": "premium",
                "subject": "Incorrect billing charges",
                "message": "I was charged twice for my subscription this month. Please refund the duplicate charge.",
                "previous_tickets": 0,
                "monthly_revenue": 299,
                "account_age_days": 90
            }
            st.session_state.pending_analysis = sample_data
            st.rerun()
        if st.button("❓ General Query"):
            sample_data = {
                "ticket_id": "GEN-003",
                "customer_tier": "free",
                "subject": "How to upgrade account",
                "message": "I want to upgrade to premium but can't find the option in my dashboard.",
                "previous_tickets": 1,
                "monthly_revenue": 0,
                "account_age_days": 30
            }
            st.session_state.pending_analysis = sample_data
            st.rerun()
        st.markdown("---")
        if st.session_state.analyzer.analysis_history:
            st.subheader("📊 Analysis History")
            st.metric("Total Analyzed", len(st.session_state.analyzer.analysis_history))
            avg_confidence = sum(a["confidence"] for a in st.session_state.analyzer.analysis_history) / len(st.session_state.analyzer.analysis_history)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["type"] == "user":
                display_chat_message(f"**New Ticket Analysis Request**\n\n{message['content']}", True)
            else:
                display_chat_message("Analysis completed! Here are the results:", False)
                display_analysis_result(message["content"])
    st.markdown("---")
    st.subheader("📝 Submit New Ticket")
    with st.form("ticket_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id = st.text_input("Ticket ID", value=f"TICKET-{len(st.session_state.messages)//2 + 1:03d}")
            customer_tier = st.selectbox("Customer Tier", ["free", "premium", "enterprise"])
            subject = st.text_input("Subject")
        with col2:
            previous_tickets = st.number_input("Previous Tickets", min_value=0, value=0)
            monthly_revenue = st.number_input("Monthly Revenue ($)", min_value=0.0, value=0.0)
            account_age_days = st.number_input("Account Age (days)", min_value=0, value=30)
        message = st.text_area("Ticket Message", height=100)
        submitted = st.form_submit_button("🚀 Analyze Ticket", use_container_width=True)
        if submitted and subject and message:
            ticket_data = {
                "ticket_id": ticket_id,
                "customer_tier": customer_tier,
                "subject": subject,
                "message": message,
                "previous_tickets": previous_tickets,
                "monthly_revenue": monthly_revenue,
                "account_age_days": account_age_days
            }
            st.session_state.pending_analysis = ticket_data
            st.rerun()
    # Process pending analysis
    if hasattr(st.session_state, 'pending_analysis'):
        ticket_data = st.session_state.pending_analysis
        del st.session_state.pending_analysis
        user_message = f"""
        **Ticket ID:** {ticket_data['ticket_id']}  
        **Subject:** {ticket_data['subject']}  
        **Customer:** {ticket_data['customer_tier'].title()} Tier  
        **Monthly Revenue:** ${ticket_data['monthly_revenue']}  
        **Message:** {ticket_data['message']}
                """
        st.session_state.messages.append({"type": "user", "content": user_message})
        with st.spinner("🔄 Analyzing ticket with multi-agent system..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(st.session_state.analyzer.analyze_ticket(ticket_data))
                loop.close()
                st.session_state.messages.append({"type": "bot", "content": result})
                
                # 🔸 Append chat history to file
                append_chat_to_file(ticket_data, result)

                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
if __name__ == "__main__":
    main()