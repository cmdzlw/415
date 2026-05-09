import re
from py2neo import Graph, Node, Relationship

# ==================== 1. 连接数据库 ====================
neo4j_url = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"  # 请根据实际密码修改
graph = Graph(neo4j_url, auth=(username, password))

# ==================== 2. 创建 30 对节点及关系 ====================
smells = Node("Smells", name="Smells directory")
s_nodes = [
    ("Bulk data transfer on slow network", 3),
    ("Data Transmission Without Compression", 4),
    ("Debuggable Release", 2),
    ("Dropped Data", 5),
    ("Durable WakeLock", 1),
    ("Early Resource Binding", 3),
    ("Inefficient Data Structure", 4),
    ("Inefficient SQL query", 5),
    ("Inefficient data format and parser", 2),
    ("Internal Getter/Setter", 1),
    ("Interrupting from background", 3),
    ("Leaking Inner Class", 4),
    ("Leaking Thread", 5),
    ("Member-Ignoring Method", 2),
    ("Nested Layout", 3),
    ("Network & IO operations in main thread", 5),
    ("No low memory resolver", 1),
    ("Not descriptive UI", 2),
    ("Overdrawn Pixel", 3),
    ("Prohibited data transfer", 4),
    ("Public data", 5),
    ("Rigid AlarmManager", 1),
    ("Set config changes", 2),
    ("Slow Loop", 3),
    ("Tracking Hardware Id", 4),
    ("Uncached Views", 5),
    ("Unclosed closable", 1),
    ("Uncontrolled focus order", 2),
    ("Unnecessary permission", 3),
    ("Untouchable", 4)
]
s_objs = []
for name, d in s_nodes:
    s = Node("Smells", name=name, dengji=d)
    s_objs.append(s)
    graph.create(s)
for s in s_objs:
    graph.create(Relationship(smells, "has_smell", s))

# 2.2 创建 Medicine 节点（解决方案），附带 solution 属性
medicine = Node("Medicine", name="Medicine directory")
m_solutions = [
    "Another approach is to give the user the choice (user preference), when high volume data transfer should be done, eg. only WiFi, 3G etc. A manual trigger of the update process might also be a solution.",
    "Compress the File object before transmitting it.",
    "Remove the attribute or set it to false explicitly",
    "The developer has to ensure that the state of the Activity or Fragment is stored, when the user inputted data. This is usually done in . It can be restored by overriding",
    "To ensure that the WakeLock will be released in all circumstances one can use the method .",
    "Move the physical resource requesting statement to the onResume().This method corresponds to the visible state of an Activity.Thus, the physical resource is consuming energy but only when the app is visible.Hence, less energy is consumed because of less time.",
    "Use SparseArray instead:\nSparseArray<Bitmap> bitmaps = new SparseArray<Bitmap>();\n bitmaps.append(i, newBitmap);",
    "According to an answer in Programmers Stackexchange the use of a SQL query is discouraged as it introduces a lot of overhead. It should be preferred to send a query to webserver and revieve e.g. a * response. This response could be compressed efficiently.\nBeyond that the projection of a query should be minimised:\nselect * from verybigtable \n select id, singlecolumn from verybigtable",
    "Use “stream” parsers instead of tree parsers\nConsider binary formats that can easily mix binary and text data into a single request",
    "Consider accessing the fields directly and only use getters and setters in public API.",
    "BroadcastRecievers and Services should not call startActivity(). Just inform the user that something happens. Use Notifications instead.",
    "Declare a static instance of the class:\nprivate static final Runnable sRunnable = new Runnable() {\npublic void run() {\n.....",
    "This will introduce a variable to check if the thread operation should still be performed.",
    "Make the method static.",
    "Nested LinearLayouts could be flattened by the use of RelativeLayouts. Or by the use of the <include>-Tag",
    "StrictMode is a developer tool which detects things you might be doing by accident and brings them to your attention so you can fix them.",
    "Use HandlerThread to perform background operations. This will help to avoid UI blocking.",
    "ERROR",
    "Use the XML attribute android:contentDescription or set it by element.setContentDescription().",
    "NONE",
    "ConnectivityManager mConnectivity;\nTelephonyManager mTelephony;\nNetworkInfo info = mConnectivity.getActiveNetworkInfo();\nif (info == null || !mConnectivity.getBackgroundDataSetting()) {\n return false;\n} ",
    "For private data use the flag Context.MODE_PRIVATE.",
    "It is recommended to use AlarmManager.setInexactRepeating(int type, long triggerAtMillis, long intervalMillis, PendingIntent operation) to ensure that the system is able to bundle several updates together.",
    "If there is a need to persist data across configuaration changes (due to orientation change, font size change) it is better to use a retained fragment.",
    "just do it",
    "The goal of the problem is to identify one concrete installation instead of a concrete hardware.",
    "You have to create a class that holds all fields of the view:ViewHolder",
    "The object should be closed properly with",
    "Use the XML attributes:",
    "The user decides wether it executes this action and by what concrete app."
]
m_names = [
    "Check network connection",
    "Add Data Compression to Apache HTTP Client based file transmission",
    "Remove Debuggable Attribute",
    "Save instance state",
    "Aquire WakeLock with timeout",
    "Move Resource Request to Visible Method",
    "Use Efficient Data Structure",
    "Use JSON query",
    "Use efficient data parser and format",
    "Direct Field Access",
    "Remove startActivity() from background",
    "Introduce Static Class",
    "Introduce Run Check Variable",
    "Introduce Static Method",
    "Flatten Layouts",
    "Use StrictMode",
    "Use HandlerThread",
    "Override onLowMemory Efficiency()",
    "Set content description",
    "Overdrawn Pixel",
    "Check background data transfer",
    "Set private mode",
    "Inexact Alarmmanager",
    "Use fragments for configuration change",
    "Enhanced For-Loop",
    "Use unique generated Id",
    "View Holder",
    "Close Closable",
    "Control focus order",
    "Use activity intent"
]
m_objs = []
for i in range(30):
    m = Node("Medicine", name=m_names[i], solution=m_solutions[i])
    m_objs.append(m)
    graph.create(m)
for m in m_objs:
    graph.create(Relationship(medicine, "MEDICINE", m))
for i in range(30):
    graph.create(Relationship(s_objs[i], "MEDICINE", m_objs[i]))

print(" 30对节点及关系已创建完毕。")


# ==================== 3. 用户输入关键词查询功能（去重输出）====================
def get_keywords_from_user():
    """获取用户输入，支持中英文逗号分隔"""
    raw = input("请输入关键词（多个用逗号分隔）：").strip()
    keywords = [kw.strip() for kw in re.split(r'[，,]', raw) if kw.strip()]
    return keywords


def search(keywords):

    for kw in keywords:
        print(f"\n 关键词：{kw}")

        # 查询匹配的 Smells 节点
        query_nodes = f"""
        MATCH (n:Smells)
        WHERE n.name CONTAINS '{kw}'
        RETURN DISTINCT n.name AS name, n.dengji AS level
        """
        nodes = graph.run(query_nodes).data()
        if nodes:
            print("  匹配的代码异味：")
            seen_nodes = set()
            for n in nodes:
                node_key = (n['name'], n['level'])
                if node_key not in seen_nodes:
                    seen_nodes.add(node_key)
                    print(f"    - {n['name']} (等级: {n['level']})")
        else:
            print("  未匹配到异味节点。")

        # 查询关联的解决方案（通过 MEDICINE 关系）
        query_solution = f"""
        MATCH (s:Smells)-[:MEDICINE]->(m:Medicine)
        WHERE s.name CONTAINS '{kw}'
        RETURN DISTINCT m.name AS medicine_name, m.solution AS solution
        """
        sols = graph.run(query_solution).data()
        if sols:
            print("  推荐解决方案：")
            seen_sols = set()
            for sol in sols:
                sol_key = sol['solution']
                if sol_key not in seen_sols:
                    seen_sols.add(sol_key)
                    print(f"    - 方案：{sol['medicine_name']}")
                    # 截断过长内容，保留前200字符
                    short_solution = sol['solution'][:200] + ('...' if len(sol['solution']) > 200 else '')
                    print(f"      详情：{short_solution}")
        else:
            print("  未找到关联的解决方案。")


if __name__ == "__main__":
    # 如果数据库已存在这些节点，可注释掉上面创建的部分（但首次运行必须创建）
    while True:
        keywords = get_keywords_from_user()
        if not keywords:
            print("未输入有效关键词，退出。")
            break
        search(keywords)
        again = input("\n是否继续查询？(y/n): ").strip().lower()
        if again != 'y':
            break
    print("程序结束。")