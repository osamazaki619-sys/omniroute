"""
Omniroute - Claude AI Integration
ملف التوصيل الشامل بين Omniroute و Claude AI مع جميع الأدوات
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import anthropic
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()


# ==================== المتغيرات والإعدادات ====================

class Config:
    """إعدادات التطبيق الشاملة"""
    
    # Claude AI Configuration
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '4096'))
    
    # Omniroute Server Configuration
    OMNIROUTE_HOST = os.getenv('OMNIROUTE_HOST', 'localhost')
    OMNIROUTE_PORT = int(os.getenv('OMNIROUTE_PORT', '20128'))
    OMNIROUTE_BASE_URL = f"http://{OMNIROUTE_HOST}:{OMNIROUTE_PORT}"
    
    # Database Configuration
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'mongodb')  # mongodb, postgresql, sqlite
    DATABASE_URL = os.getenv('DATABASE_URL', 'mongodb://localhost:27017/omniroute')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'omniroute_db')
    
    # API Keys للأدوات الأخرى
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY', '')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
    
    # Authentication
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'omniroute.log')
    
    # Server Configuration
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    TIMEOUT = int(os.getenv('TIMEOUT', '30'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))


# ==================== Enums ====================

class ToolType(Enum):
    """أنواع الأدوات المتاحة"""
    CLAUDE = "claude"
    OPENAI = "openai"
    SEARCH = "search"
    DATABASE = "database"
    RETRIEVAL = "retrieval"
    CACHE = "cache"


class RequestStatus(Enum):
    """حالات الطلبات"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ==================== Data Classes ====================

@dataclass
class Message:
    """بيانات الرسالة"""
    role: str  # user, assistant, system
    content: str
    timestamp: str = None
    metadata: Dict = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.metadata:
            self.metadata = {}


@dataclass
class ToolResponse:
    """استجابة الأداة"""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ConversationContext:
    """سياق المحادثة"""
    session_id: str
    messages: List[Message]
    tools_used: List[str]
    metadata: Dict
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


# ==================== Claude AI Integration ====================

class ClaudeAIIntegration:
    """فئة التوصيل مع Claude AI"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)
        self.model = Config.CLAUDE_MODEL
        self.max_tokens = Config.CLAUDE_MAX_TOKENS
    
    def send_message(
        self, 
        messages: List[Dict], 
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> Dict:
        """
        إرسال رسالة إلى Claude API
        
        Args:
            messages: قائمة الرسائل
            system_prompt: رسالة النظام
            tools: قائمة الأدوات المتاحة
            
        Returns:
            استجابة Claude
        """
        try:
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            if tools:
                params["tools"] = tools
            
            response = self.client.messages.create(**params)
            
            return {
                "success": True,
                "content": response.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_tool_use(self, response: Dict) -> List[ToolResponse]:
        """
        معالجة استخدام الأدوات من استجابة Claude
        
        Args:
            response: استجابة Claude
            
        Returns:
            قائمة استجابات الأدوات
        """
        tool_responses = []
        
        for content_block in response.get("content", []):
            if content_block.get("type") == "tool_use":
                tool_name = content_block.get("name")
                tool_input = content_block.get("input", {})
                
                # تنفيذ الأداة
                result = self.execute_tool(tool_name, tool_input)
                tool_responses.append(result)
        
        return tool_responses
    
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResponse:
        """
        تنفيذ أداة محددة
        
        Args:
            tool_name: اسم الأداة
            tool_input: مدخلات الأداة
            
        Returns:
            استجابة الأداة
        """
        import time
        start_time = time.time()
        
        try:
            if tool_name == "search":
                result = self.search_tool(tool_input)
            elif tool_name == "database":
                result = self.database_tool(tool_input)
            elif tool_name == "retrieval":
                result = self.retrieval_tool(tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            
            execution_time = time.time() - start_time
            
            return ToolResponse(
                tool_name=tool_name,
                success=result.get("success", False),
                data=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResponse(
                tool_name=tool_name,
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    @staticmethod
    def search_tool(tool_input: Dict) -> Dict:
        """أداة البحث"""
        query = tool_input.get("query", "")
        
        try:
            # استخدام SerpAPI للبحث
            if Config.SERPAPI_API_KEY:
                url = "https://serpapi.com/search"
                params = {
                    "q": query,
                    "api_key": Config.SERPAPI_API_KEY
                }
                response = requests.get(url, params=params, timeout=Config.TIMEOUT)
                data = response.json()
                
                return {
                    "success": True,
                    "query": query,
                    "results": data.get("organic_results", [])
                }
            else:
                return {
                    "success": False,
                    "error": "SerpAPI key not configured"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def database_tool(tool_input: Dict) -> Dict:
        """أداة قاعدة البيانات"""
        operation = tool_input.get("operation")  # query, insert, update, delete
        collection = tool_input.get("collection")
        data = tool_input.get("data", {})
        
        try:
            if Config.DATABASE_TYPE == "mongodb":
                from pymongo import MongoClient
                client = MongoClient(Config.DATABASE_URL)
                db = client[Config.DATABASE_NAME]
                collection_obj = db[collection]
                
                if operation == "query":
                    result = list(collection_obj.find(data, {"_id": 0}))
                elif operation == "insert":
                    result = collection_obj.insert_one(data)
                    result = {"inserted_id": str(result.inserted_id)}
                elif operation == "update":
                    filter_query = data.get("filter", {})
                    update_data = data.get("update", {})
                    result = collection_obj.update_one(filter_query, {"$set": update_data})
                    result = {"modified_count": result.modified_count}
                else:
                    result = {"error": f"Unknown operation: {operation}"}
                
                return {
                    "success": True,
                    "operation": operation,
                    "result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"Database type {Config.DATABASE_TYPE} not yet implemented"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def retrieval_tool(tool_input: Dict) -> Dict:
        """أداة الاسترجاع (Retrieval)"""
        query = tool_input.get("query", "")
        
        try:
            if Config.PINECONE_API_KEY:
                # مثال على استخدام Pinecone
                return {
                    "success": True,
                    "query": query,
                    "results": []  # نتائج البحث من Pinecone
                }
            else:
                return {
                    "success": False,
                    "error": "Pinecone API key not configured"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# ==================== Omniroute Integration ====================

class OmnirouteIntegration:
    """فئة التوصيل مع Omniroute Server"""
    
    def __init__(self):
        self.base_url = Config.OMNIROUTE_BASE_URL
        self.timeout = Config.TIMEOUT
        self.max_retries = Config.MAX_RETRIES
    
    def health_check(self) -> bool:
        """التحقق من صحة الخادم"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def send_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        إرسال طلب إلى Omniroute
        
        Args:
            endpoint: نقطة النهاية
            method: طريقة HTTP
            data: البيانات المرسلة
            params: معاملات الاستعلام
            
        Returns:
            استجابة Omniroute
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                if method == "POST":
                    response = requests.post(
                        url,
                        json=data,
                        headers=headers,
                        timeout=self.timeout,
                        params=params
                    )
                elif method == "GET":
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        params=params
                    )
                else:
                    return {"success": False, "error": f"Unsupported method: {method}"}
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Status code: {response.status_code}",
                        "response": response.text
                    }
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    continue
                return {"success": False, "error": "Request timeout"}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    continue
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}


# ==================== Main Integration Manager ====================

class OmnirouteClaudeManager:
    """مدير التوصيل الرئيسي بين Omniroute و Claude"""
    
    def __init__(self):
        self.claude = ClaudeAIIntegration()
        self.omniroute = OmnirouteIntegration()
        self.conversations: Dict[str, ConversationContext] = {}
    
    def create_conversation(self, session_id: str) -> ConversationContext:
        """إنشاء محادثة جديدة"""
        context = ConversationContext(
            session_id=session_id,
            messages=[],
            tools_used=[],
            metadata={}
        )
        self.conversations[session_id] = context
        return context
    
    def add_message(self, session_id: str, message: Message) -> bool:
        """إضافة رسالة إلى المحادثة"""
        if session_id not in self.conversations:
            self.create_conversation(session_id)
        
        self.conversations[session_id].messages.append(message)
        self.conversations[session_id].updated_at = datetime.now().isoformat()
        return True
    
    def process_with_claude(
        self,
        session_id: str,
        user_input: str,
        system_prompt: Optional[str] = None,
        available_tools: Optional[List[Dict]] = None
    ) -> Dict:
        """
        معالجة مدخل المستخدم مع Claude
        
        Args:
            session_id: معرف الجلسة
            user_input: مدخل المستخدم
            system_prompt: رسالة النظام
            available_tools: الأدوات المتاحة
            
        Returns:
            استجابة معالجة كاملة
        """
        # إضافة رسالة المستخدم
        user_message = Message(role="user", content=user_input)
        self.add_message(session_id, user_message)
        
        # تحضير الرسائل للإرسال
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversations[session_id].messages
        ]
        
        # إرسال إلى Claude
        claude_response = self.claude.send_message(
            messages=messages,
            system_prompt=system_prompt,
            tools=available_tools
        )
        
        if not claude_response.get("success"):
            return {
                "success": False,
                "error": claude_response.get("error"),
                "session_id": session_id
            }
        
        # معالجة استخدام الأدوات إن وجدت
        tool_responses = self.claude.process_tool_use(claude_response)
        
        # إضافة استجابة Claude
        assistant_message = Message(
            role="assistant",
            content=json.dumps(claude_response.get("content", [])),
            metadata={"tools_used": len(tool_responses)}
        )
        self.add_message(session_id, assistant_message)
        
        return {
            "success": True,
            "session_id": session_id,
            "response": claude_response,
            "tool_responses": [
                {
                    "tool": tr.tool_name,
                    "success": tr.success,
                    "execution_time": tr.execution_time
                }
                for tr in tool_responses
            ],
            "message_count": len(self.conversations[session_id].messages)
        }
    
    def get_conversation_history(self, session_id: str) -> Optional[List[Dict]]:
        """الحصول على سجل المحادثة"""
        if session_id not in self.conversations:
            return None
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.conversations[session_id].messages
        ]
    
    def export_conversation(self, session_id: str, format: str = "json") -> Optional[str]:
        """تصدير المحادثة"""
        if session_id not in self.conversations:
            return None
        
        context = self.conversations[session_id]
        
        if format == "json":
            return json.dumps({
                "session_id": context.session_id,
                "created_at": context.created_at,
                "updated_at": context.updated_at,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp
                    }
                    for msg in context.messages
                ]
            }, ensure_ascii=False, indent=2)
        
        return None


# ==================== Environment Variables Template ====================

def generate_env_template() -> str:
    """إنشاء قالب متغيرات البيئة"""
    return """
# ====== Claude AI Configuration ======
CLAUDE_API_KEY=your-claude-api-key
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_MAX_TOKENS=4096

# ====== Omniroute Configuration ======
OMNIROUTE_HOST=localhost
OMNIROUTE_PORT=20128

# ====== Database Configuration ======
DATABASE_TYPE=mongodb
DATABASE_URL=mongodb://localhost:27017/omniroute
DATABASE_NAME=omniroute_db

# ====== Third Party API Keys ======
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
SERPAPI_API_KEY=your-serpapi-api-key
PINECONE_API_KEY=your-pinecone-api-key

# ====== Authentication ======
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256

# ====== Logging ======
LOG_LEVEL=INFO
LOG_FILE=omniroute.log

# ====== Server Configuration ======
DEBUG=False
TIMEOUT=30
MAX_RETRIES=3
"""


# ==================== Example Usage ====================

def main():
    """مثال على الاستخدام"""
    
    # التحقق من الاتصال بـ Omniroute
    manager = OmnirouteClaudeManager()
    
    if not manager.omniroute.health_check():
        print("⚠️ Warning: Omniroute server is not responding")
    else:
        print("✅ Omniroute server is healthy")
    
    # إنشاء محادثة
    session_id = "session_001"
    manager.create_conversation(session_id)
    
    # معالجة سؤال المستخدم
    user_query = "ما هو الذكاء الاصطناعي؟"
    
    response = manager.process_with_claude(
        session_id=session_id,
        user_input=user_query,
        system_prompt="أنت مساعد ذكي يجيب على الأسئلة باللغة العربية."
    )
    
    print("\n" + "="*50)
    print("Response:", json.dumps(response, ensure_ascii=False, indent=2))
    print("="*50)
    
    # عرض سجل المحادثة
    history = manager.get_conversation_history(session_id)
    print("\nConversation History:")
    for msg in history:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")


if __name__ == "__main__":
    # طباعة قالب متغيرات البيئة
    print("📋 Environment Variables Template:")
    print(generate_env_template())
    
    # تشغيل المثال
    print("\n🚀 Running integration example...\n")
    main()
