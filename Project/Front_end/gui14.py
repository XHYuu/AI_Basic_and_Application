import json
import locale
import re
import sys
import time

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QGraphicsOpacityEffect, QTextEdit
)
from PySide6.QtGui import QFont, QFontMetrics, QPixmap, QKeyEvent, QPainter, QPainterPath
from PySide6.QtCore import Qt, QTimer, QProcess
from openai import OpenAI


# 自定义输入框：回车发送，Shift+回车换行
class MyTextEdit(QTextEdit):
    def __init__(self, parent=None, send_callback=None):
        super().__init__(parent)
        self.send_callback = send_callback

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() == Qt.ShiftModifier:
                super().keyPressEvent(event)  # Shift+Enter换行
            else:
                if self.send_callback:
                    self.send_callback()
        else:
            super().keyPressEvent(event)


class ChatBox(QWidget):
    def __init__(self):
        super().__init__()
        self.waiting_for_confirm = False  # 只在结果判定后才True
        # self.setStyleSheet("background-color: white;")
        self.initUI()
        self.model_process = None
        self.question = None

    def initUI(self):
        self.setWindowTitle('AI health-consult Chatbot')
        self.setGeometry(100, 100, 800, 700)

        main_layout = QVBoxLayout()

        title_layout = QHBoxLayout()
        icon_label = QLabel()
        icon_pixmap = QPixmap("image/gpt_icon.png")
        size = 32  # 圆角图标的尺寸
        icon_pixmap = icon_pixmap.scaled(size, size)  # 先缩放到合适大小

        # 创建圆角蒙版
        rounded = QPixmap(size, size)
        rounded.fill(Qt.transparent)  # 透明底

        painter = QPainter(rounded)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, size, size, 10, 10)  # 8是圆角半径，可自调
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, icon_pixmap)
        painter.end()

        icon_label.setPixmap(rounded)
        icon_label.setFixedSize(40, 40)  # 根据需要调整大小
        icon_label.setScaledContents(True)  # 图标自适应大小

        title_text = QLabel("Question consultation assistant")
        title_text.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title_layout.addStretch()  # 左侧弹性空间，实现整体居中
        title_layout.addWidget(icon_label)
        title_layout.addSpacing(8)  # 图标和文字之间留点空隙
        title_layout.addWidget(title_text)
        title_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
                    QScrollArea {
                        border: none;            /* 去掉外框线 */
                    }
                """)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.chat_container = QWidget()

        # # ==== 背景图 start ====
        # logo_label = QLabel(self.chat_container)
        # logo_pixmap = QPixmap("hku_logo.png")
        # logo_label.setPixmap(logo_pixmap)
        # logo_label.setScaledContents(True)
        # logo_label.setGeometry(0, 0, 600, 400)  # 初始大小
        # logo_label.lower()  # 放到底层
        #
        # opacity_effect = QGraphicsOpacityEffect()
        # opacity_effect.setOpacity(0.5)  # 50%透明
        # logo_label.setGraphicsEffect(opacity_effect)
        #
        # def resize_logo():
        #     logo_label.setGeometry(0, 0, self.chat_container.width(), self.chat_container.height())
        #
        # self.chat_container.resizeEvent = lambda event: resize_logo()
        # # ==== 背景图 end ====

        self.chat_layout = QVBoxLayout()
        self.chat_container.setLayout(self.chat_layout)
        scroll_area.setWidget(self.chat_container)

        input_layout = QHBoxLayout()
        self.input_box = MyTextEdit(send_callback=self.send_message)
        self.input_box.setStyleSheet("""
            QTextEdit {
                border: none;            /* 去掉外框线 */
                border-radius: 12px;     /* 圆角半径，自行调整 */
                background: #f5f5f5;     /* 可选，淡灰底色更好看 */
                padding: 8px;            /* 可选，内部留白 */
                font-size: 16px;         /* 可选，字体大小 */
            }
        """)
        self.input_box.setFixedHeight(50)
        self.send_button = QPushButton('Send')
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;        /* 绿色 */
                color: white;                     /* 字体白色 */
                border: 2px solid #4CAF50;        /* 绿色边框 */
                border-radius: 18px;              /* 圆角，数值越大越圆 */
                padding: 8px 24px;                /* 内边距决定按钮大小 */
                font-size: 16px;                  /* 字体大小可选 */
            }
            QPushButton:hover {
                background-color: #43a047;        /* 悬浮时更深的绿色 */
                border: 2px solid #388e3c;
            }
            QPushButton:pressed {
                background-color: #388e3c;        /* 按下时更深的绿色 */
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_button)

        main_layout.addLayout(title_layout)
        main_layout.addWidget(scroll_area)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

        # 系统首次欢迎语
        self.add_message("Welcome to Health Consultation. Please enter your question.", is_sender=False)
        self.add_message("Note! This result is for reference only.", is_sender=False)

    def send_message(self):
        message = self.input_box.toPlainText().strip()
        if not message:
            return
        self.add_message(message, is_sender=True)
        self.input_box.clear()

        # 只在等待确认时处理“是/否”
        if self.waiting_for_confirm:
            if message == "yes" or message == "Yes":
                self.add_message("Thanks.", is_sender=False)
                self.waiting_for_confirm = False
            elif message == "no" or message == "No":
                self.add_message(
                    "I'm very sorry. I will give you more related information. ",
                    is_sender=False)
                if self.question is not None:
                    self.add_more_info(self.question)
                time.sleep(4)
                self.add_message(
                    "Do you have any other questions?",
                    is_sender=False)

                self.waiting_for_confirm = False
            else:
                self.add_message("By default, your question has been resolved. Please ask a new one.", is_sender=False)
                self.waiting_for_confirm = False
            return

        self.add_message("Thinking...", is_sender=False)
        self.start_model_process(message)

    def add_message(self, message, is_sender=False):
        message_label = QLabel(message)
        message_label.setWordWrap(False)
        font = QFont("Times New Roman", 14)
        message_label.setFont(font)
        message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_max_width = 380
        message_label_max_width = text_max_width + 50
        message_label.setMaximumWidth(message_label_max_width)
        font_metrics = QFontMetrics(font)
        elided_text = self.wrap_text(message, font, text_max_width)
        message_label.setText(elided_text)
        message_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        message_layout = QHBoxLayout()

        if is_sender:
            message_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            message_label.setStyleSheet("""
                QLabel {
                    background-color: #dcf8c6;
                    color: #333;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px;
                }
            """)
            message_layout.addStretch()
            message_layout.addWidget(message_label)
        else:
            message_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            message_label.setStyleSheet("""
                QLabel {
                    background-color: #ffffff;
                    color: #333;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px;
                }
            """)
            message_layout.addWidget(message_label)
            message_layout.addStretch()
        self.chat_layout.addLayout(message_layout)
        self.chat_layout.addStretch()
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        scroll_bar = self.findChild(QScrollArea).verticalScrollBar()
        QTimer.singleShot(100, lambda: scroll_bar.setValue(scroll_bar.maximum()))

    def wrap_text(self, message, font, max_width):
        font_metrics = QFontMetrics(font)
        lines = []
        current_line = ""

        tokens = re.findall(r'\S+|\s+|[.,;:!?]', message)

        for token in tokens:
            if '-' in token and not token.strip() in "-":
                parts = token.split('-')
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        part += '-'

                    test_line = current_line + part
                    text_line_width = font_metrics.horizontalAdvance(test_line)

                    if text_line_width > max_width:
                        if current_line.strip():
                            lines.append(current_line.strip())
                        current_line = part.lstrip()
                    else:
                        current_line = test_line
            else:
                test_line = current_line + token
                text_line_width = font_metrics.horizontalAdvance(test_line)

                if text_line_width > max_width:
                    if current_line.strip():  # 当前行有内容
                        lines.append(current_line.strip())
                    current_line = token.lstrip()  # 新的一行从当前token开始，去除前置空格
                else:
                    current_line = test_line

        if current_line.strip():
            lines.append(current_line.strip())

        wrapped_text = "\r\n".join(lines)
        return wrapped_text

    def start_model_process(self, user_input):
        if self.model_process:
            self.model_process.kill()
            self.model_process = None

        try:
            self.model_process = QProcess(self)
            self.model_process.setProgram(sys.executable)
            self.model_process.setArguments(['mock_model.py'])

            model_handle = {
                "user_input": user_input,
            }
            input_str = json.dumps(model_handle)

            self.model_process.readyReadStandardOutput.connect(self.handle_model_output)
            self.model_process.started.connect(lambda: self.model_process.write((input_str + '\n').encode('utf-8')))
            self.model_process.start()
        except Exception as e:
            print(e)

    def handle_model_output(self):
        if not self.model_process:
            return

        output = bytes(self.model_process.readAllStandardOutput()).decode(locale.getpreferredencoding(),
                                                                          errors='replace')
        ask_confirm = False
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    self.question = data["user_input"]
                except json.JSONDecodeError:
                    self.add_message(line.strip(), is_sender=False)
                    if "The result is" in line:
                        ask_confirm = True
        if ask_confirm:
            self.add_message(
                "Have your questions been resolved? If not, please reply 'No'. "
                "We will provide more possible solutions.",
                is_sender=False)
            self.waiting_for_confirm = True

    def add_more_info(self, message):
        KIMI_API_KEY = "sk-rgxAPpDWCjSjvhdwvr7khgETRTzKWFqDZGS1FbFx8l11BVoq"

        client = OpenAI(
            api_key=KIMI_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
        print(message)

        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system",
                 "content": "你要扮演一名医疗助手，你需要根据患者的症状描述提供回答，包括但不限于：病情诊断、追问更多细节、提供健康建议。请用英文回答。"},
                {"role": "user",
                 "content": message}
            ],
            temperature=0.3,
        )
        answer = completion.choices[0].message.content
        answer = answer.replace("I'm not a doctor, but ", "")

        self.add_message(
            answer,
            is_sender=False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    chat_box = ChatBox()
    chat_box.show()
    sys.exit(app.exec())
