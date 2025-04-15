import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

base_url = "http://" + os.environ.get('OLLAMA_HOST', "localhost") + ":11434"
model = 'qwen2.5-coder:1.5b'

SQL_SYSTEM_PROMPT = """You are an agent designed to generating SQL queries.
                    You work with one and ONLY ONE TABLE NAMED ```drawing```.
                    All the generated SQL query must START WITH ```SELECT``` ONLY.
                    DO NOT make any DML statements with (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE etc.).
                    Following the DEFAULT_SQL_EXAMPLES format to generate only 
                    the appropriate SQL query for the user's ```question```.
                    DO NOT use ```*``` BUT use the column names in the table schema.
                    DO NOT include any other column names that are not in the table schema.
                    DO NOT include explanations or additional information.
                    DO NOT pretty print the query.
                    ALWAYS use alias names in the SQL query when using function.
                    RETURN the SQL query in a single line without any carriage returns.
                    RETURN the SQL query with ```;``` at the end.
                    TRANSLATE the alias names in the SQL to the same language as the ```question```.
                    DO NOT use any other language than the ```question``` language.
                    IMPORTANT:
                    If the ```question``` doesn't seem related to the database, just RETURN ```Error```."""

DEFAULT_SQL_EXAMPLES = """
                    Examples:
                    Question: How many drawing numbers are there?
                    SQL Query: SELECT COUNT(*) FROM drawing;
                    Question: Count the total occurrences of '佐井鋼㈱' in material supplier.
                    SQL Query: SELECT COUNT(*) FROM drawing WHERE material_sup = '佐井鋼㈱';
                    Question: Give me the details for the drawing number '<drawing_number>'.
                    SQL Query: SELECT * FROM drawing WHERE drawing_number = '<drawing_number>';
                    Question: Show all unique material suppliers.
                    SQL Query: SELECT DISTINCT material_sup FROM drawing;
                    Question: What is the total other cost?
                    SQL Query: SELECT SUM(other_cost) FROM drawing;
                    Question: What is the other cost of the drawing number '<drawing_number>'?
                    SQL Query: SELECT other_cost FROM drawing WHERE drawing_number = '<drawing_number>';
                    
                    Question: 図面番号はいくつありますか？
                    SQL Query: SELECT COUNT(*) FROM drawing;
                    Question: '佐井鋼㈱'が材料サプライヤーに出現した総回数を数えてください。
                    SQL Query: SELECT COUNT(*) FROM drawing WHERE material_sup = '佐井鋼㈱';
                    Question: 図面番号「<drawing_number>」の詳細をすべて教えていただけますか？
                    SQL Query: SELECT * FROM drawing WHERE drawing_number = '<drawing_number>';
                    Question: すべてのユニークな材料サプライヤーを表示してください。
                    SQL Query: SELECT DISTINCT material_sup FROM drawing;
                    Question: その他の費用の合計はいくらですか？
                    SQL Query: SELECT SUM(other_cost) FROM drawing;
                    Question: 図面番号「<drawing_number>」のその他の費用はいくらですか？
                    SQL Query: SELECT other_cost FROM drawing WHERE drawing_number = '<drawing_number>';
                    
                    Question: 有多少个图纸编号？
                    SQL Query: SELECT COUNT(*) FROM drawing;
                    Question: 统计'佐井鋼㈱'在材料供应商中的总出现次数。
                    SQL Query: SELECT COUNT(*) FROM drawing WHERE material_sup = '佐井鋼㈱';
                    Question: 请提供图纸编号'<drawing_number>'的详细信息。
                    SQL Query: SELECT * FROM drawing WHERE drawing_number = '<drawing_number>';
                    Question: 显示所有唯一的材料供应商。
                    SQL Query: SELECT DISTINCT material_sup FROM drawing;
                    Question: 其他成本的总和是多少？
                    SQL Query: SELECT SUM(other_cost) FROM drawing;
                    Question: 图纸编号 '<drawing_number>' 的其他成本是多少？
                    SQL Query: SELECT other_cost FROM drawing WHERE drawing_number = '<drawing_number>';
                    
                    Question: Có bao nhiêu số bản vẽ?
                    SQL Query: SELECT COUNT(*) FROM drawing;
                    Question: Đếm tổng số lần xuất hiện của '佐井鋼㈱' trong nhà cung cấp vật liệu.
                    SQL Query: SELECT COUNT(*) FROM drawing WHERE material_sup = '佐井鋼㈱';
                    Question: Cho tôi biết chi tiết về số bản vẽ '<drawing_number>'.
                    SQL Query: SELECT * FROM drawing WHERE drawing_number = '<drawing_number>';
                    Question: Hiển thị tất cả các nhà cung cấp vật liệu duy nhất.
                    SQL Query: SELECT DISTINCT material_sup FROM drawing;
                    Question: Tổng chi phí khác là bao nhiêu?
                    SQL Query: SELECT SUM(other_cost) FROM drawing;
                    uestion: Chi phí khác của bản vẽ số '<drawing_number>' là bao nhiêu?
                    SQL Query: SELECT other_cost FROM drawing WHERE drawing_number = '<drawing_number>';
                    """
                    
column_name_mapping = {
    "general_cost": {"ja": "一般費用", "zh": "一般成本", "vi": "chi phí chung"},
    "grinding_cost": {"ja": "研削費用", "zh": "磨削成本", "vi": "chi phí mài"},
    "heat_treatment_cost": {"ja": "熱処理費用", "zh": "热处理成本", "vi": "chi phí xử lý nhiệt"},
    "lathe_cost": {"ja": "旋盤費用", "zh": "车床成本", "vi": "chi phí tiện"},
    "material_cost": {"ja": "材料費用", "zh": "材料成本", "vi": "chi phí vật liệu"},
    "milling_cost": {"ja": "フライス加工費用", "zh": "铣削成本", "vi": "chi phí phay"},
    "other_cost": {"ja": "その他の費用", "zh": "其他成本", "vi": "chi phí khác"},
    "selling_price": {"ja": "販売価格", "zh": "销售价格", "vi": "giá bán"},
    "transportation_cost": {"ja": "輸送費用", "zh": "运输成本", "vi": "chi phí vận chuyển"},
    "welding_cost": {"ja": "溶接費用", "zh": "焊接成本", "vi": "chi phí hàn"},
    "defect_details": {"ja": "欠陥の詳細", "zh": "缺陷详情", "vi": "chi tiết khuyết điểm"},
    "drawing_number": {"ja": "図面番号", "zh": "图纸编号", "vi": "số bản vẽ"},
    "name": {"ja": "名前", "zh": "姓名", "vi": "tên"},
    "general_sup": {"ja": "一般サプライヤー", "zh": "一般供应商", "vi": "nhà cung cấp chung"},
    "grinding_sup": {"ja": "研削サプライヤー", "zh": "磨削供应商", "vi": "nhà cung cấp mài"},
    "heat_treatment_sup": {"ja": "熱処理サプライヤー", "zh": "热处理供应商", "vi": "nhà cung cấp xử lý nhiệt"},
    "lathe_sup": {"ja": "旋盤サプライヤー", "zh": "车床供应商", "vi": "nhà cung cấp tiện"},
    "material_sup": {"ja": "材料サプライヤー", "zh": "材料供应商", "vi": "nhà cung cấp vật liệu"},
    "milling_sup": {"ja": "フライス加工サプライヤー", "zh": "铣削供应商", "vi": "nhà cung cấp phay"},
    "other_sup": {"ja": "その他のサプライヤー", "zh": "其他供应商", "vi": "nhà cung cấp khác"},
    "transportation_sup": {"ja": "輸送サプライヤー", "zh": "运输供应商", "vi": "nhà cung cấp vận chuyển"},
    "welding_sup": {"ja": "溶接サプライヤー", "zh": "焊接供应商", "vi": "nhà cung cấp hàn"}
}

DEFAULT_PROMPT_TEMPLATE = """
                        ### Table Schema ###
                        {context}
                        
                        ### Question ###
                        {question}
                        
                        ### SQL Query ###
                        """

def create_chain() -> ChatPromptTemplate:
    
    print("Start chat")
    system = SystemMessagePromptTemplate.from_template(SQL_SYSTEM_PROMPT)
    example = SystemMessagePromptTemplate.from_template(DEFAULT_SQL_EXAMPLES)
    prompt = HumanMessagePromptTemplate.from_template(DEFAULT_PROMPT_TEMPLATE)
    
    messages = [system, example, prompt]
    template = ChatPromptTemplate(messages)
    
    # Chain setup - set temperature lower for more deterministic SQL
    llm = ChatOllama(base_url=base_url, model=model, temperature=0.1, keep_alive="24h")
    
    return template | llm | StrOutputParser()

def translate_columns(question, column_mapping):
    for eng_col, translations in column_mapping.items():
        for lang, translated_name in translations.items():
            if translated_name in question:
                question = question.replace(translated_name, eng_col)
    return question

def chat(prompt: str, table: str) -> str:

    try:
        prompt = translate_columns(prompt, column_name_mapping)
        qna_chain = create_chain()
        
        result = qna_chain.invoke({
            'context': table, 
            'question': prompt
        })
        result = result.strip()
        print(f"Result: {result}")
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"































