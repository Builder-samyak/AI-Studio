"""
app/streamlit_app.py — RadiScan FINAL
Destroyers | 42174 AI Studio Autumn 2026
Compact hospital-grade Radiology PMS
"""
import os, sys, datetime, io, random, smtplib, copy
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from src.nlp.nlp_assistant import RadiScanAssistant
    _nlp = RadiScanAssistant(api_key=os.getenv("OPENAI_API_KEY",""))
    def ask_assistant(q,p,r,h): return _nlp.ask(q,p,r,h)
    def quick_qs(r):             return _nlp.get_quick_questions(r)
except ImportError:
    def ask_assistant(q,p,r,h): return "NLP module not found. Place nlp_assistant.py in src/nlp/"
    def quick_qs(r):             return ["Why was this classified this way?","What are the next steps?"]

MODEL_PATH   = os.getenv("MODEL_PATH",
    "/home/sagemaker-user/user-default-efs/destroyers_model/efficientnet_b0_best.pth")
THRESHOLD    = 0.4
IMG_MEAN     = [0.485, 0.456, 0.406]
IMG_STD      = [0.229, 0.224, 0.225]
PAGE_SIZE    = 12

DEFAULT_USERS = {
    "admin": {
        "password":"password123",
        "name":"Dr. Harshitha Kolgatta Swamy",
        "title":"Consultant Radiologist",
        "email":"harshitha.kolgattaswamy@student.uts.edu.au",
    },
    "dr.aagusthya": {
        "password":"password123",
        "name":"Dr. Aagusthya Shanker",
        "title":"Senior Radiologist",
        "email":"AagusthyaShanker@student.uts.edu.au",
    },
    "dr.samyak": {
        "password":"password123",
        "name":"Dr. Samyak Borkar",
        "title":"Radiology Registrar",
        "email":"Samyak.Borkar@student.uts.edu.au",
    },
}

# Email settings for forgot-password OTP.
# Recommended: set these as environment variables instead of hardcoding them.
# PowerShell example:
#   $env:RADISCAN_SMTP_EMAIL="yourgmail@gmail.com"
#   $env:RADISCAN_SMTP_PASSWORD="your_gmail_app_password"
SMTP_SERVER = os.getenv("RADISCAN_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("RADISCAN_SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("RADISCAN_SMTP_EMAIL", "02.harshitha@gmail.com")
SMTP_PASSWORD = os.getenv("RADISCAN_SMTP_PASSWORD", "YOUR_GMAIL_APP_PASSWORD")
OTP_EXPIRY_MINUTES = 10

st.set_page_config(
    page_title="RadiScan — AI Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;font-size:14px;}
h1,h2,h3,h4,h5,h6{font-family:'Inter',sans-serif;}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
.block-container{
    padding-top:1.4rem !important;
    padding-bottom:1rem !important;
}
section[data-testid="stMain"]>div:first-child{
    padding-top:0.8rem !important;
}
div[data-testid="stVerticalBlock"]:first-child{
    margin-top:0.5rem !important;
}

/* ── Login ── */
.login-wrap{max-width:400px;margin:3rem auto 0;background:white;border-radius:12px;
    padding:2.25rem 2.5rem;box-shadow:0 4px 24px rgba(0,0,0,0.10);
    border:1px solid #e2e8f0;}
.login-logo{text-align:center;margin-bottom:1.5rem;}
.login-logo h1{font-size:1.6rem;font-weight:700;color:#0f172a;margin:0.25rem 0 0.15rem;}
.login-logo p{color:#64748b;font-size:0.82rem;margin:0;}

/* ── Header bar ── */
.main-header{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
    padding:1rem 1.5rem;border-radius:10px;margin-bottom:1rem;}
.main-header h1{color:white;margin:0;font-size:1.4rem;font-weight:700;}
.main-header p{color:#94a3b8;margin:0;font-size:0.78rem;}

/* ── Patient cards ── */
.patient-card{background:white;border:1px solid #e2e8f0;border-radius:8px;
    padding:0.75rem 1rem;margin-bottom:0.4rem;}
.patient-card.urgent {border-left:3px solid #ef4444;}
.patient-card.review {border-left:3px solid #f59e0b;}
.patient-card.clear  {border-left:3px solid #10b981;}
.patient-card.pending{border-left:3px solid #6366f1;}
.patient-card.discharged{border-left:3px solid #8b5cf6;}
.patient-card.new-pt{border-left:3px solid #0ea5e9;background:#f0f9ff;}

/* ── Badges ── */
.b-urgent   {background:#fef2f2;color:#dc2626;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}
.b-review   {background:#fffbeb;color:#d97706;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}
.b-clear    {background:#f0fdf4;color:#059669;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}
.b-pending  {background:#eef2ff;color:#4f46e5;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}
.b-discharged{background:#f5f3ff;color:#6d28d9;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}
.b-new      {background:#e0f2fe;color:#0369a1;padding:2px 7px;border-radius:20px;font-size:0.68rem;font-weight:700;}

/* ── Results ── */
.result-cancer {background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:0.85rem;}
.result-clear  {background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:0.85rem;}

/* ── Metrics ── */
.metric-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;
    padding:0.6rem;text-align:center;}
.metric-box .val{font-size:1.2rem;font-weight:700;color:#0f172a;}
.metric-box .lbl{font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;}

/* ── Chat ── */
.chat-user{background:#eff6ff;border-radius:10px 10px 3px 10px;
    padding:0.55rem 0.8rem;margin:0.3rem 0;margin-left:12%;font-size:0.82rem;}
.chat-bot {background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px 10px 10px 3px;
    padding:0.55rem 0.8rem;margin:0.3rem 0;margin-right:12%;font-size:0.82rem;}
.chat-scroll{max-height:260px;overflow-y:auto;padding:0.4rem;}

/* ── Section title ── */
.sec{font-size:0.7rem;font-weight:600;color:#64748b;text-transform:uppercase;
    letter-spacing:0.08em;margin-bottom:0.6rem;padding-bottom:0.3rem;
    border-bottom:1px solid #e2e8f0;}

/* ── Disclaimer ── */
.disclaimer{background:#fefce8;border:1px solid #fde047;border-radius:7px;
    padding:0.55rem 0.85rem;font-size:0.73rem;color:#854d0e;margin-top:0.75rem;}

/* ── Flag / notes boxes ── */
.flag-box {background:#fef2f2;border:1px solid #fecaca;border-radius:7px;
    padding:0.55rem;margin-top:0.4rem;font-size:0.78rem;}
.notes-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:7px;
    padding:0.55rem;font-size:0.78rem;color:#374151;}
.status-box{border-radius:6px;padding:0.4rem 0.7rem;font-size:0.8rem;
    font-weight:600;text-align:center;margin-bottom:0.6rem;}

/* ── Scan history ── */
.scan-item{background:#f8fafc;border-radius:6px;padding:0.4rem 0.6rem;
    margin-bottom:0.3rem;font-size:0.76rem;}

/* ── Inline nav buttons (Back + Logout on patient/dashboard pages) ── */
.nav-bar{display:flex;gap:0.5rem;margin-bottom:0.6rem;align-items:center;}
.nav-bar .stButton>button{
    background:#1e3a5f !important;color:white !important;
    border:1px solid #334155 !important;font-size:0.8rem !important;
    padding:0.3rem 0.85rem !important;border-radius:5px !important;}
.nav-bar .stButton>button:hover{background:#2563eb !important;}
button[data-testid="baseButton-secondary"][key*="dash_logout"],
button[key="dash_logout"]{
    background:#1e3a5f !important;color:white !important;
    border:1px solid #334155 !important;font-size:0.8rem !important;}

/* ── Sidebar ── */
div[data-testid="stSidebar"]{background:#0f172a;}
div[data-testid="stSidebar"] *{color:white !important;}
div[data-testid="stSidebar"] hr{border-color:#1e293b;}
div[data-testid="stSidebar"] .stButton button{
    background:#1e3a5f;border:1px solid #334155;color:white;width:100%;font-size:0.82rem;}
div[data-testid="stSidebar"] .stButton button:hover{background:#2563eb;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_new(p):
    try:
        d = datetime.datetime.strptime(p.get("registered_date",""), "%d %b %Y").date()
        return (datetime.date.today() - d).days <= 3
    except: return False

def _badge(status, new=False):
    if new: return "<span class='b-new'>NEW</span>"
    return {
        "urgent":    "<span class='b-urgent'>🚨 URGENT</span>",
        "review":    "<span class='b-review'>⚠ REVIEW</span>",
        "clear":     "<span class='b-clear'>✓ CLEAR</span>",
        "pending":   "<span class='b-pending'>○ PENDING</span>",
        "discharged":"<span class='b-discharged'>🏥 DISCHARGED</span>",
    }.get(status, f"<span class='b-pending'>{status.upper()}</span>")

def _next_pid(pts):
    nums = [int(p.get("patient_id","PAT-2026-0000").split("-")[-1])
            for p in pts.values() if p.get("patient_id","").count("-")==2]
    return f"PAT-2026-{(max(nums)+1 if nums else 1):04d}"


def _email_config_ready():
    return (
        SMTP_EMAIL
        and SMTP_PASSWORD
        and SMTP_EMAIL != "YOUR_GMAIL@gmail.com"
        and SMTP_PASSWORD != "YOUR_GMAIL_APP_PASSWORD"
    )


def send_reset_email(receiver_email, otp_code):
    if not _email_config_ready():
        st.error(
            "Email is not configured. Set RADISCAN_SMTP_EMAIL and "
            "RADISCAN_SMTP_PASSWORD, then restart Streamlit."
        )
        return False

    try:
        subject = "RadiScan Password Reset Code"
        body = f"""Hello,

Your RadiScan password reset verification code is:

{otp_code}

This code will expire in {OTP_EXPIRY_MINUTES} minutes.

If you did not request this reset, please ignore this email.

RadiScan Security System
"""

        msg = MIMEMultipart()
        msg["From"] = SMTP_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True

    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False


def _clear_reset_state():
    st.session_state.reset_user = None
    st.session_state.reset_otp = None
    st.session_state.reset_otp_created = None
    st.session_state.otp_verified = False
    st.session_state.otp_email = None


# ── Patient data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_patients():
    random.seed(42)
    names = [
        "Margaret Chen","Priya Sharma","Sandra O'Brien","Fatima Al-Hassan",
        "Helen Kowalski","Diane Foster","Ruth Mitchell","Angela Torres",
        "Yuki Nakamura","Ingrid Hansen","Amara Osei","Claudia Reyes",
        "Patricia Walsh","Nadia Petrov","Mei Lin Tan","Brigitte Moreau",
        "Vera Hoffman","Aigerim Bekova","Siobhan Murphy","Leila Ahmadi",
        "Grace Okafor","Carmen Vega","Sonia Kapoor","Elena Popescu",
        "Frances Dubois","Hana Suzuki","Rosa Martinez","Celine Fontaine",
        "Adaeze Nnaji","Mira Andersen","Beatrice Larsson","Zara Ahmed",
        "Lena Wagner","Nour Khalil","Yewande Adeyemi","Sofia Petersen",
        "Imelda Cruz","Astrid Lindqvist","Rania Hassan","Olga Kovalenko",
        "Thandeka Dlamini","Isabelle Bernard","Mei Fong Lim","Catalina Ruiz",
        "Nadege Dupont","Wanjiru Kamau","Tatiana Sorokina","Kiri Waititi",
        "Amelia Johansson","Lakshmi Patel",
    ]
    statuses = (["urgent"]*8+["review"]*10+["clear"]*18+["pending"]*8+["discharged"]*6)
    random.shuffle(statuses)
    complaints = [
        "Abnormal mammogram — irregular mass, left breast",
        "Routine annual screening — no symptoms",
        "Post-treatment monitoring — lumpectomy 18 months ago",
        "Self-detected lump — upper outer quadrant, right breast",
        "Annual screening — HRT patient",
        "Pre-surgical assessment — lumpectomy scheduled",
        "Family history of breast cancer — enhanced screening",
        "Nipple discharge — unilateral, spontaneous",
        "Skin changes over left breast — dimpling noted",
        "Post-mastectomy surveillance — 2 years post-op",
    ]
    histories = [
        "Family history of breast cancer (mother). BRCA1 negative.",
        "No family history. Non-smoker. BMI 24.",
        "Stage II IDC treated with lumpectomy + radiation. Tamoxifen ongoing.",
        "First presentation. No prior imaging. BRCA status unknown.",
        "HRT for 4 years. Increased screening protocol.",
        "BRCA2 positive. IDC Grade 2 confirmed on core biopsy.",
        "Hypertension, well-controlled. Prior fibroadenoma excision 2019.",
    ]
    referring = [
        "Dr. Sarah Mitchell (GP)","Dr. Alan Roberts (GP)",
        "Dr. Wei Zhang (Oncology)","Dr. Nancy Park (Breast Surgery)",
        "Dr. Tom Fletcher (GP)","Dr. James Okafor (Oncology)",
    ]
    allergies = ["NKDA","Penicillin","Sulfa drugs","Iodine contrast","NKDA","NKDA"]
    priorities = ["Routine","High","Urgent","Routine","Routine"]
    base = datetime.date(2026,5,22)
    pts = {}
    for i,name in enumerate(names):
        pid = f"P{i+1:03d}"
        st_ = statuses[i]
        age = random.randint(35,76)
        dob_y = 2026-age
        days_ago = random.randint(0,14)
        reg_days = random.randint(0,30)
        vd = (base - datetime.timedelta(days=days_ago)).strftime("%d %b %Y")
        rd = (base - datetime.timedelta(days=reg_days)).strftime("%d %b %Y")
        nd = (base + datetime.timedelta(days=random.randint(14,60))).strftime("%d %b %Y")
        if st_=="urgent":    ap,ac=round(random.uniform(0.80,0.97),2),"Cancer"
        elif st_=="review":  ap,ac=round(random.uniform(0.50,0.79),2),"Cancer"
        elif st_=="clear":   ap,ac=round(random.uniform(0.04,0.25),2),"Non-Cancer"
        elif st_=="discharged":ap,ac=round(random.uniform(0.03,0.15),2),"Non-Cancer"
        else: ap,ac=None,None
        cs=None
        if ac:
            h_=random.randint(8,17);m_=random.randint(0,59)
            cs={"class":ac,"probability":ap,"performed":f"{vd} {h_:02d}:{m_:02d}"}
        ft=fto=None
        if st_=="urgent":
            fto=random.choice(["Dr. James Okafor (Oncology)","Dr. Nancy Park (Surgery)"])
            ft=vd
        prior=[]
        if st_!="pending":
            for _ in range(random.randint(1,2)):
                pd_=(base-datetime.timedelta(days=random.randint(180,730))).strftime("%d %b %Y")
                res,rs_=random.choice([
                    ("BIRADS 1 — negative","clear"),("BIRADS 2 — benign","clear"),
                    ("BIRADS 3 — probably benign","clear"),("BIRADS 4B — suspicious","review"),
                    ("No evidence of recurrence","clear")])
                prior.append({"date":pd_,"type":random.choice(["Mammogram","MRI","Histopathology"]),"result":res,"status":rs_})
        fn=name.split()[0].lower(); ln=name.split()[-1].lower()
        pts[pid]={
            "id":pid,"patient_id":f"PAT-2026-{i+1:04d}","name":name,"age":age,"sex":"Female",
            "dob":f"{random.randint(1,28)} {random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])} {dob_y}",
            "mrn":f"MRN-{random.randint(2022,2026)}-{random.randint(1000,9999)}",
            "phone":f"+61 4{random.randint(10,99)} {random.randint(100,999)} {random.randint(100,999)}",
            "email":f"{fn}.{ln}@email.com",
            "medicare":f"{random.randint(1000,9999)} {random.randint(10000,99999)} {random.randint(0,9)}",
            "allergies":random.choice(allergies),
            "emergency_contact":f"+61 4{random.randint(10,99)} {random.randint(100,999)} {random.randint(100,999)}",
            "priority":random.choice(priorities),
            "last_visit":vd,"next_visit":nd,"registered_date":rd,
            "referring":random.choice(referring),
            "complaint":random.choice(complaints),"history":random.choice(histories),
            "status":st_,"prior_scans":prior,"ai_result":cs,
            "flagged_to":fto,"flag_time":ft,"notes":"",
        }
    real=[
        {"id":"P001","patient_id":"PAT-2026-0001","name":"Margaret Chen","age":52,"sex":"Female",
         "dob":"12 Mar 1972","mrn":"MRN-2024-0891","phone":"+61 412 345 678",
         "email":"margaret.chen@email.com","medicare":"2345 67890 1",
         "allergies":"Penicillin","emergency_contact":"David Chen: +61 413 456 789",
         "priority":"Urgent","last_visit":"21 May 2026","next_visit":"4 Jun 2026",
         "registered_date":"20 May 2026","referring":"Dr. Sarah Mitchell (GP)",
         "complaint":"Abnormal mammogram — irregular mass, left breast",
         "history":"Family history of breast cancer (mother). Annual screening. BRCA1 negative.",
         "status":"urgent","prior_scans":[
             {"date":"14 Jan 2026","type":"Mammogram","result":"BIRADS 4B — suspicious","status":"review"},
             {"date":"10 Jan 2025","type":"Mammogram","result":"BIRADS 3 — probably benign","status":"clear"},
         ],
         "ai_result":{"class":"Cancer","probability":0.87,"performed":"21 May 2026 09:14"},
         "flagged_to":"Dr. James Okafor (Oncology)","flag_time":"21 May 2026 09:45",
         "notes":"Urgent referral placed. Biopsy scheduled 28 May."},
        {"id":"P002","patient_id":"PAT-2026-0002","name":"Priya Sharma","age":45,"sex":"Female",
         "dob":"3 Aug 1980","mrn":"MRN-2024-0445","phone":"+61 421 987 654",
         "email":"priya.sharma@email.com","medicare":"3456 78901 2",
         "allergies":"NKDA","emergency_contact":"Raj Sharma: +61 422 876 543",
         "priority":"Routine","last_visit":"20 May 2026","next_visit":"20 Nov 2026",
         "registered_date":"19 May 2026","referring":"Dr. Alan Roberts (GP)",
         "complaint":"Routine annual screening — no symptoms",
         "history":"No family history. Non-smoker. BMI 24.",
         "status":"clear","prior_scans":[
             {"date":"18 May 2025","type":"Mammogram","result":"BIRADS 1 — negative","status":"clear"},
         ],
         "ai_result":{"class":"Non-Cancer","probability":0.12,"performed":"20 May 2026 14:22"},
         "flagged_to":None,"flag_time":None,"notes":"Normal screening. Follow-up 6 months."},
        {"id":"P003","patient_id":"PAT-2026-0003","name":"Sandra O'Brien","age":61,"sex":"Female",
         "dob":"29 Nov 1964","mrn":"MRN-2024-1203","phone":"+61 433 111 222",
         "email":"sandra.obrien@email.com","medicare":"4567 89012 3",
         "allergies":"Sulfa drugs","emergency_contact":"Michael O'Brien: +61 434 222 333",
         "priority":"High","last_visit":"19 May 2026","next_visit":"2 Jun 2026",
         "registered_date":"18 May 2026","referring":"Dr. Wei Zhang (Oncology)",
         "complaint":"Post-treatment monitoring — lumpectomy 18 months ago",
         "history":"Stage II IDC. Lumpectomy + radiation. Tamoxifen ongoing. BRCA1/2 negative.",
         "status":"review","prior_scans":[
             {"date":"5 Dec 2025","type":"MRI","result":"No evidence of recurrence","status":"clear"},
             {"date":"8 Jun 2025","type":"Histopathology","result":"Non-malignant — post-treatment","status":"clear"},
         ],
         "ai_result":{"class":"Cancer","probability":0.63,"performed":"19 May 2026 11:05"},
         "flagged_to":"Dr. Wei Zhang (Oncology)","flag_time":"19 May 2026 11:30",
         "notes":"Moderate confidence. MRI correlation requested."},
        {"id":"P004","patient_id":"PAT-2026-0004","name":"Fatima Al-Hassan","age":38,"sex":"Female",
         "dob":"17 Apr 1988","mrn":"MRN-2025-0067","phone":"+61 444 555 666",
         "email":"fatima.alhassan@email.com","medicare":"5678 90123 4",
         "allergies":"NKDA","emergency_contact":"Ahmed Al-Hassan: +61 445 666 777",
         "priority":"Routine","last_visit":"22 May 2026","next_visit":"22 Aug 2026",
         "registered_date":"22 May 2026","referring":"Dr. Sarah Mitchell (GP)",
         "complaint":"Self-detected lump — upper outer quadrant, right breast",
         "history":"First presentation. No prior imaging. BRCA status unknown.",
         "status":"pending","prior_scans":[],"ai_result":None,
         "flagged_to":None,"flag_time":None,"notes":""},
        {"id":"P005","patient_id":"PAT-2026-0005","name":"Helen Kowalski","age":57,"sex":"Female",
         "dob":"2 Feb 1969","mrn":"MRN-2024-0788","phone":"+61 455 777 888",
         "email":"helen.kowalski@email.com","medicare":"6789 01234 5",
         "allergies":"NKDA","emergency_contact":"Jan Kowalski: +61 456 888 999",
         "priority":"Routine","last_visit":"18 May 2026","next_visit":"18 Nov 2026",
         "registered_date":"17 May 2026","referring":"Dr. Tom Fletcher (GP)",
         "complaint":"Annual screening — HRT patient",
         "history":"HRT for 4 years. Increased screening protocol. Benign cyst aspirated 2022.",
         "status":"clear","prior_scans":[
             {"date":"22 May 2025","type":"Mammogram","result":"BIRADS 2 — benign","status":"clear"},
         ],
         "ai_result":{"class":"Non-Cancer","probability":0.08,"performed":"18 May 2026 10:44"},
         "flagged_to":None,"flag_time":None,"notes":"Clear. Continue annual screening."},
        {"id":"P006","patient_id":"PAT-2026-0006","name":"Diane Foster","age":49,"sex":"Female",
         "dob":"8 Sep 1976","mrn":"MRN-2025-0312","phone":"+61 466 999 000",
         "email":"diane.foster@email.com","medicare":"7890 12345 6",
         "allergies":"Iodine contrast","emergency_contact":"Robert Foster: +61 467 000 111",
         "priority":"Urgent","last_visit":"21 May 2026","next_visit":"4 Jun 2026",
         "registered_date":"21 May 2026","referring":"Dr. Nancy Park (Breast Surgery)",
         "complaint":"Pre-surgical assessment — lumpectomy scheduled next month",
         "history":"BRCA2 positive. IDC Grade 2 confirmed on core biopsy Apr 2026.",
         "status":"urgent","prior_scans":[
             {"date":"10 Apr 2026","type":"Core biopsy","result":"IDC Grade 2 confirmed","status":"review"},
             {"date":"3 Mar 2026","type":"MRI","result":"3.2cm lesion, upper inner quadrant","status":"review"},
         ],
         "ai_result":{"class":"Cancer","probability":0.91,"performed":"21 May 2026 15:33"},
         "flagged_to":"Dr. Nancy Park (Surgery) + Dr. James Okafor (Oncology)",
         "flag_time":"21 May 2026 15:55","notes":"Very high confidence. MDT meeting requested."},
    ]
    for r in real:
        pts[r["id"]] = r
    return pts


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        from torchvision import models as tvm
        import torch.nn as nn
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(p=0.3,inplace=True),
                                     nn.Linear(m.classifier[1].in_features,2))
        if os.path.exists(MODEL_PATH):
            ck = torch.load(MODEL_PATH, map_location=dev)
            m.load_state_dict(ck["model_state_dict"])
        m.to(dev).eval(); return m,dev,None
    except Exception as e:
        return None,"cpu",str(e)

def _preprocess(img):
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((128,128)), transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)])
    return t(img.convert("RGB")).unsqueeze(0)

def _inference(model, img_t, device):
    with torch.no_grad():
        prob = torch.softmax(model(img_t.to(device)),dim=1)[0,1].item()
    return ("Cancer" if prob>=THRESHOLD else "Non-Cancer"), prob

def _gradcam(model, img_t, device):
    try:
        import cv2
        grads,acts=[],[]
        def _hk(m,i,o):
            acts.append(o.detach().cpu())
            o.register_hook(lambda g: grads.append(g.detach().cpu()))
        h = model.features[8].register_forward_hook(_hk)
        model.eval()
        out = model(img_t.to(device))
        model.zero_grad(); out[0,out.argmax(1).item()].backward(); h.remove()
        if not grads or not acts: return None
        g=grads[0].squeeze().numpy(); a=acts[0].squeeze().numpy()
        if g.ndim<3: return None
        w=g.mean(axis=(1,2))
        cam=np.zeros(a.shape[1:],dtype=np.float32)
        for wi,ai in zip(w,a): cam+=wi*ai
        cam=np.maximum(cam,0)
        if cam.max()==0: return None
        cam=cv2.resize(cam,(128,128)); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        heat=cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
        inp=img_t.squeeze().numpy().transpose(1,2,0)
        inp=np.clip(inp*np.array(IMG_STD)+np.array(IMG_MEAN),0,1)
        bgr=cv2.cvtColor((inp*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
        ov=cv2.addWeighted(bgr,0.55,heat,0.45,0)
        return cv2.cvtColor(ov,cv2.COLOR_BGR2RGB)
    except: return None

def _generate_pdf(p,ai_result,notes,rad_comment=""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,HRFlowable
    from reportlab.lib.styles import ParagraphStyle
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,topMargin=1.8*cm,bottomMargin=1.8*cm,
                          leftMargin=2*cm,rightMargin=2*cm)
    W=17.2*cm
    def ps(name,**kw): return ParagraphStyle(name,**kw)
    S={"t":ps("T",fontSize=15,fontName="Helvetica-Bold",textColor=colors.HexColor("#0f172a"),spaceAfter=2),
       "s":ps("S",fontSize=8,fontName="Helvetica",textColor=colors.HexColor("#64748b"),spaceAfter=8),
       "h":ps("H",fontSize=10,fontName="Helvetica-Bold",textColor=colors.HexColor("#0f172a"),spaceAfter=3,spaceBefore=8),
       "b":ps("B",fontSize=9,fontName="Helvetica",textColor=colors.HexColor("#334155"),spaceAfter=3),
       "r":ps("R",fontSize=13,fontName="Helvetica-Bold",spaceAfter=4),
       "d":ps("D",fontSize=7.5,fontName="Helvetica",textColor=colors.HexColor("#94a3b8")),
       "i":ps("I",fontSize=9,fontName="Helvetica-Oblique",textColor=colors.HexColor("#334155"),spaceAfter=3)}
    def tbl(data,cw):
        t=Table(data,colWidths=cw)
        t.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
            ("LINEBELOW",(0,0),(-1,-2),0.3,colors.HexColor("#e2e8f0")),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#d1d5db"))]))
        return t
    now=datetime.datetime.now().strftime("%d %b %Y %H:%M")
    story=[
        Spacer(1, 0.15*cm),
        Paragraph("RadiScan — Clinical Radiology Report", S["t"]),
        Spacer(1, 0.12*cm),
        Paragraph(
            "AI-Assisted Breast Cancer Detection System  │  "
            "Team Destroyers  │  42174 AI Studio  │  "
            "University of Technology Sydney", S["s"]),
        Spacer(1, 0.15*cm),
        HRFlowable(width=W, thickness=2, color=colors.HexColor("#0f172a")),
        Spacer(1, 0.3*cm),
    ]
    cw=[3.5*cm,5*cm,3.2*cm,5.5*cm]
    demo=[
        ["Patient Name",p["name"],"Patient ID",p.get("patient_id","—")],
        ["MRN",p["mrn"],"Medicare",p.get("medicare","—")],
        ["Age / Sex",f"{p['age']} / {p['sex']}","Date of Birth",p["dob"]],
        ["Phone",p.get("phone","—"),"Email",p.get("email","—")],
        ["Last Visit",p["last_visit"],"Next Visit",p["next_visit"]],
        ["Referring",p["referring"],"Report Date",now],
        ["Priority",p.get("priority","—"),"Allergies",p.get("allergies","—")],
        ["Status",p["status"].upper(),"Registered",p.get("registered_date","—")],
    ]
    story+=[tbl(demo,cw),Spacer(1,0.1*cm)]
    ec=Table([["Emergency Contact",p.get("emergency_contact","—")]],colWidths=[3.5*cm,13.7*cm])
    ec.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,0),"Helvetica-Bold"),("FONTNAME",(1,0),(1,0),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),6),
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f8fafc")),("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#d1d5db"))]))
    story+=[ec, Spacer(1, 0.35*cm)]
    story+=[
        Paragraph("Chief Complaint", S["h"]), Spacer(1, 0.05*cm),
        Paragraph(p["complaint"], S["b"]), Spacer(1, 0.1*cm),
        Paragraph("Medical History", S["h"]), Spacer(1, 0.05*cm),
        Paragraph(p["history"], S["b"]),
    ]
    if ai_result:
        pred=ai_result["class"]; prob=ai_result["probability"]
        col="#dc2626" if pred=="Cancer" else "#16a34a"
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph("AI Classification Result", S["h"]))
        story.append(Spacer(1, 0.08*cm))
        story.append(Paragraph(
            f"<font color='{col}'><b>{'⚠  MALIGNANT' if pred=='Cancer' else '✓  NON-MALIGNANT'}</b></font>",
            S["r"]))
        story.append(Spacer(1, 0.1*cm))
        story.append(tbl([
            ["Malignant Probability",f"{prob*100:.1f}%","Threshold","40%"],
            ["Model","EfficientNet-B0 v0.2","Analysed",ai_result.get("performed","—")],
            ["Accuracy","91.6%","Recall","89.3%"],["AUC","0.9700","GPU","Tesla T4 (AWS SageMaker)"]],
            [4*cm,4.6*cm,3.5*cm,5.1*cm]))
    if p.get("flagged_to"):
        story+=[Paragraph("Referral",S["h"]),
                Paragraph(f"Referred to: {p['flagged_to']}",S["b"]),
                Paragraph(f"Time: {p.get('flag_time','—')}",S["b"])]
    if notes.strip():
        story+=[Paragraph("Clinical Notes",S["h"]),Paragraph(notes,S["b"])]
    if rad_comment.strip():
        story+=[Paragraph("Radiologist Sign-off",S["h"]),Paragraph(rad_comment,S["i"])]
    from reportlab.lib.styles import ParagraphStyle as PS
    story+=[Spacer(1,0.4*cm),HRFlowable(width=W,thickness=0.5,color=colors.HexColor("#e2e8f0")),
            Spacer(1,0.1*cm),
            Paragraph("DISCLAIMER: RadiScan is an AI decision-support system (EfficientNet-B0 v0.2). "
                      "Not a clinical diagnosis. Requires validation by a qualified medical professional. "
                      "Team Destroyers | 42174 AI Studio | UTS.",S["d"])]
    doc.build(story); buf.seek(0); return buf.read()


# ── Auto-draft helpers ─────────────────────────────────────────────────────────
def _auto_draft_note(pred, prob, patient):
    now  = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    hist = patient.get("history","")
    pct  = f"{prob*100:.1f}%"
    if pred == "Cancer" and prob >= 0.8:
        return (
            f"[DRAFT — {now}] AI histopathology analysis: highly suspicious for malignancy "
            f"(EfficientNet-B0, {pct}). Features consistent with IDC: nuclear atypia, "
            f"raised N:C ratio, disordered glandular architecture. Grad-CAM localises region "
            f"of interest. Recommend: urgent MDT referral, core needle biopsy with ER/PR/HER2 "
            f"receptor panel. History: {hist}"
        )
    elif pred == "Cancer":
        return (
            f"[DRAFT — {now}] AI analysis: moderate-confidence malignant features ({pct}). "
            f"Cellular patterns consistent with IDC — confidence below high-threshold. "
            f"Recommend: compare with prior imaging, consider MRI/ultrasound, MDT discussion "
            f"before biopsy escalation. History: {hist}"
        )
    return (
        f"[DRAFT — {now}] AI analysis: no malignant features ({pct}, below 40% threshold). "
        f"Histomorphology consistent with non-malignant tissue. "
        f"Continue standard screening protocol. Clinical correlation required. "
        f"History: {hist}"
    )


def _auto_draft_signoff(pred, prob):
    pct = f"{prob*100:.1f}%"
    if pred == "Cancer" and prob >= 0.8:
        return (
            f"AI-assisted histopathology: highly suspicious for malignancy ({pct}). "
            f"Urgent oncology referral and tissue diagnosis correlation recommended."
        )
    elif pred == "Cancer":
        return (
            f"AI-assisted analysis: suspicious for malignancy ({pct}). "
            f"MDT review and biopsy correlation recommended before definitive management."
        )
    return (
        f"AI-assisted analysis: no malignant features identified ({pct}). "
        f"Continue surveillance per protocol. Clinical correlation required."
    )


# ── Session ────────────────────────────────────────────────────────────────────
for k,v in {
    "logged_in":False,"user":None,"selected_patient":None,"patients":None,
    "chat":{},"scan_results":{},"notifications":[],"page":0,
    "login_mode":"login","reset_user":None,
    "reset_otp":None,"reset_otp_created":None,"otp_verified":False,"otp_email":None,
    "users":None,
}.items():
    if k not in st.session_state:
        st.session_state[k]=copy.deepcopy(v)

if st.session_state.users is None:
    st.session_state.users = copy.deepcopy(DEFAULT_USERS)
USERS = st.session_state.users

if st.session_state.patients is None:
    st.session_state.patients = get_patients()
patients = st.session_state.patients
model, device, model_error = load_model()
nlp_ok = bool(os.getenv("OPENAI_API_KEY",""))


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    _,col,_ = st.columns([1,1.1,1])
    with col:
        st.markdown("""
        <div class='login-wrap'>
          <div class='login-logo'>
            <div style='font-size:2.2rem;'>🔬</div>
            <h1>RadiScan</h1>
            <p>AI-Assisted Breast Cancer Detection System</p>
            <p style='font-size:0.72rem;color:#94a3b8;margin-top:0.3rem;'>
              Radiology Department · 42174 AI Studio · UTS</p>
          </div>
        </div>""", unsafe_allow_html=True)

        mode = st.session_state.login_mode
        if mode == "login":
            with st.form("lf"):
                usr = st.text_input("Username", placeholder="Enter username")
                pwd = st.text_input("Password", type="password", placeholder="Enter password")
                sub = st.form_submit_button("Sign In →", type="primary", use_container_width=True)
            if sub:
                if usr in USERS and USERS[usr]["password"]==pwd:
                    st.session_state.logged_in=True; st.session_state.user=USERS[usr]
                    st.session_state.login_mode="login"; st.rerun()
                else: st.error("Invalid username or password.")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Forgot password?", use_container_width=True):
                st.session_state.login_mode="forgot"; st.rerun()

        elif mode == "forgot":
            st.markdown("**Reset Password**")
            st.caption("Enter your username and registered UTS email to receive a verification code.")
            fp_u = st.text_input("Username", key="fp_u")
            fp_e = st.text_input("Registered email", placeholder="your@student.uts.edu.au", key="fp_e")

            if st.button("Send Verification Code", type="primary", use_container_width=True):
                fp_u_clean = fp_u.strip()
                fp_e_clean = fp_e.strip()

                if fp_u_clean in USERS and USERS[fp_u_clean]["email"].lower() == fp_e_clean.lower():
                    otp = str(random.randint(100000, 999999))
                    st.session_state.reset_otp = otp
                    st.session_state.reset_otp_created = datetime.datetime.now()
                    st.session_state.reset_user = fp_u_clean
                    st.session_state.otp_email = fp_e_clean
                    st.session_state.otp_verified = False

                    if send_reset_email(fp_e_clean, otp):
                        st.success("Verification code sent to your email.")
                        st.session_state.login_mode = "verify_otp"
                        st.rerun()
                else:
                    st.error("Username and email do not match.")

            if st.button("← Back to login", use_container_width=True):
                _clear_reset_state()
                st.session_state.login_mode = "login"
                st.rerun()

        elif mode == "verify_otp":
            st.markdown("**Email Verification**")
            st.caption(f"Verification code sent to {st.session_state.otp_email}")
            entered_otp = st.text_input("Enter 6-digit verification code", max_chars=6, key="entered_otp")

            if st.button("Verify Code", type="primary", use_container_width=True):
                created = st.session_state.reset_otp_created
                expired = True
                if created is not None:
                    expired = datetime.datetime.now() > created + datetime.timedelta(minutes=OTP_EXPIRY_MINUTES)

                if expired:
                    st.error("Verification code expired. Please request a new code.")
                    _clear_reset_state()
                    st.session_state.login_mode = "forgot"
                    st.rerun()
                elif entered_otp.strip() == st.session_state.reset_otp:
                    st.session_state.otp_verified = True
                    st.session_state.login_mode = "reset"
                    st.success("Code verified successfully.")
                    st.rerun()
                else:
                    st.error("Invalid verification code.")

            if st.button("← Back", use_container_width=True):
                st.session_state.login_mode = "forgot"
                st.rerun()

        elif mode == "reset":
            ru = st.session_state.reset_user or ""

            if not st.session_state.otp_verified:
                st.warning("OTP verification required.")
                st.session_state.login_mode = "forgot"
                st.rerun()

            if ru in USERS:
                st.success(f"Verified: {USERS[ru]['name']}")

            np_ = st.text_input("New password", type="password", placeholder="Minimum 8 characters", key="rp_n")
            cp_ = st.text_input("Confirm password", type="password", key="rp_c")

            if st.button("Update Password", type="primary", use_container_width=True):
                if ru not in USERS:
                    st.error("Reset session expired. Please try again.")
                    _clear_reset_state()
                    st.session_state.login_mode = "forgot"
                    st.rerun()
                elif len(np_) < 8:
                    st.error("Minimum 8 characters required.")
                elif np_ != cp_:
                    st.error("Passwords do not match.")
                else:
                    USERS[ru]["password"] = np_
                    st.session_state.users = USERS
                    _clear_reset_state()
                    st.success("Password updated successfully. Please log in with your new password.")
                    st.session_state.login_mode = "login"
                    st.rerun()

            if st.button("← Back to login", key="rb2", use_container_width=True):
                _clear_reset_state()
                st.session_state.login_mode = "login"
                st.rerun()

        st.markdown(
            "<div style='text-align:center;margin-top:0.8rem;font-size:0.72rem;color:#94a3b8;'>"
            "For authorised radiology personnel only. Research prototype — not for clinical diagnosis.</div>",
            unsafe_allow_html=True)
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
u = st.session_state.user
with st.sidebar:
    st.markdown("### 🔬 RadiScan")
    st.markdown(f"**{u['name']}**")
    st.markdown(f"*{u['title']}*")
    st.markdown("---")
    urgent_n = sum(1 for p in patients.values() if p["status"]=="urgent")
    review_n = sum(1 for p in patients.values() if p["status"]=="review")
    new_n    = sum(1 for p in patients.values() if _is_new(p))
    st.markdown(f"🚨 **{urgent_n}** urgent cases")
    st.markdown(f"⚠ **{review_n}** for review")
    if new_n: st.markdown(f"🆕 **{new_n}** new patients")
    st.markdown("---")
    st.markdown("**Model:** EfficientNet-B0 v0.2")
    st.markdown("**Accuracy:** 91.6%  |  **AUC:** 0.970")
    st.markdown(f"**NLP:** GPT-4o {'✅' if nlp_ok else '⚠ key not set'}")
    st.markdown("---")
    if st.session_state.selected_patient:
        if st.button("← Patient List"):
            st.session_state.selected_patient=None; st.session_state.page=0; st.rerun()
    if st.button("🚪 Sign Out"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()
    if st.session_state.notifications:
        st.markdown("---")
        st.markdown("**📬 Notifications**")
        for n in st.session_state.notifications[-3:]:
            st.markdown(f"<small>✉ {n}</small>", unsafe_allow_html=True)

badges = {
    "urgent":"<span class='b-urgent'>🚨 URGENT</span>",
    "review":"<span class='b-review'>⚠ REVIEW</span>",
    "clear": "<span class='b-clear'>✓ CLEAR</span>",
    "pending":"<span class='b-pending'>○ PENDING</span>",
    "discharged":"<span class='b-discharged'>🏥 DISCHARGED</span>",
}
status_col={"urgent":"#ef4444","review":"#f59e0b","clear":"#10b981","pending":"#6366f1","discharged":"#8b5cf6"}


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT LIST
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.selected_patient is None:

    # Top bar: header + logout inline
    _wh1, _wh2 = st.columns([6, 1])
    with _wh1:
        st.markdown(f"""
        <div class='main-header' style='margin-bottom:0.65rem;'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <div>
              <h1>🔬 Patient Worklist</h1>
              <p>AI-Assisted Breast Cancer Detection · Radiology Department · 42174 AI Studio</p>
            </div>
            <div style='font-size:0.8rem;color:#94a3b8;text-align:right;'>
              <div style='color:white;font-weight:600;'>{u['name']}</div>
              <div>{u['title']}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    with _wh2:
        st.markdown("<div style='padding-top:0.55rem;'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="dash_logout", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Toolbar
    c1,c2,c3,c4 = st.columns([3,1.2,1,0.9])
    with c1: search = st.text_input("🔍 Search",placeholder="Name, Patient ID, MRN, complaint...",label_visibility="collapsed")
    with c2: filt   = st.selectbox("",["All","Urgent","Review","Pending","Clear","Discharged","New (≤3 days)"],label_visibility="collapsed")
    with c3: sort   = st.selectbox("",["Newest first","Urgency","Name A–Z"],label_visibility="collapsed")
    with c4:
        if st.button("＋ New Patient",use_container_width=True,type="primary"):
            st.session_state.selected_patient="__new__"; st.rerun()

    # Filter + sort
    order={"urgent":0,"review":1,"pending":2,"clear":3,"discharged":4}
    rows = list(patients.values())
    if search:
        s=search.lower()
        rows=[p for p in rows if s in p["name"].lower() or s in p.get("mrn","").lower()
              or s in p.get("patient_id","").lower() or s in p.get("complaint","").lower()]
    if filt=="New (≤3 days)": rows=[p for p in rows if _is_new(p)]
    elif filt!="All": rows=[p for p in rows if p["status"]==filt.lower()]

    def _rts(p):
        try: return datetime.datetime.strptime(p.get("registered_date","01 Jan 2000"),"%d %b %Y")
        except: return datetime.datetime.min
    if sort=="Newest first": rows=sorted(rows,key=_rts,reverse=True)
    elif sort=="Urgency": rows=sorted(rows,key=lambda p:order.get(p["status"],5))
    elif sort=="Name A–Z": rows=sorted(rows,key=lambda p:p["name"])

    total=len(rows); pages=max(1,(total+PAGE_SIZE-1)//PAGE_SIZE)
    pg=min(st.session_state.page,pages-1)
    page_rows=rows[pg*PAGE_SIZE:(pg+1)*PAGE_SIZE]
    st.markdown(f"<div style='font-size:0.78rem;color:#64748b;margin:0.3rem 0 0.5rem;'>"
                f"{total} patient(s) · Page {pg+1} of {pages}</div>",unsafe_allow_html=True)

    for p in page_rows:
        is_n = _is_new(p)
        badge_html = _badge(p["status"],is_n)
        row_css = "new-pt" if is_n else p["status"]
        ai_html=""
        if p.get("ai_result"):
            ar=p["ai_result"]; cl=ar["class"]; pr=ar["probability"]
            clr="#dc2626" if cl=="Cancer" else "#059669"
            ai_html=(f" · AI: <b style='color:{clr};'>{cl}</b> ({pr*100:.0f}%)")

        c1,c2 = st.columns([5.8,0.7])
        with c1:
            st.markdown(f"""
            <div class='patient-card {row_css}'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-weight:600;color:#0f172a;'>{p["name"]}</span>
                <span style='color:#94a3b8;font-size:0.78rem;'>{p.get("patient_id","")}</span>
                <span style='color:#64748b;font-size:0.78rem;'>{p["mrn"]}</span>
                {badge_html}
              </div>
              <div style='margin-top:0.2rem;font-size:0.78rem;color:#475569;'>
                Age {p["age"]} · {p["last_visit"]} · {p["referring"].split("(")[0].strip()}
                {ai_html} · {p["complaint"][:60]}...
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open →",key=f"o_{p['id']}",use_container_width=True):
                st.session_state.selected_patient=p["id"]
                if p["id"] not in st.session_state.chat:
                    st.session_state.chat[p["id"]]=[]
                st.rerun()

    if pages>1:
        pc1,pc2,pc3=st.columns([1,2,1])
        with pc1:
            if pg>0:
                if st.button("← Prev",use_container_width=True):
                    st.session_state.page=pg-1; st.rerun()
        with pc2:
            st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:#64748b;padding:0.4rem;'>"
                        f"Page {pg+1} / {pages}</div>",unsafe_allow_html=True)
        with pc3:
            if pg<pages-1:
                if st.button("Next →",use_container_width=True):
                    st.session_state.page=pg+1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# NEW PATIENT REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.selected_patient == "__new__":

    st.markdown("""
    <div class='main-header'>
      <h1>👤 Register New Patient</h1>
      <p>Complete all required fields and click Register</p>
    </div>""", unsafe_allow_html=True)

    if st.button("← Back to Patient List"):
        st.session_state.selected_patient=None; st.rerun()

    with st.form("new_pt"):
        st.markdown("<div class='sec'>Patient Identity</div>",unsafe_allow_html=True)
        f1,f2,f3=st.columns(3)
        with f1:
            np_name=st.text_input("Full Name *")
            np_pid=st.text_input("Patient ID *",value=_next_pid(patients))
            np_mrn=st.text_input("MRN *",placeholder="MRN-2026-XXXX")
            np_dob=st.text_input("Date of Birth",placeholder="DD Mon YYYY")
        with f2:
            np_age=st.number_input("Age *",18,110,40)
            np_sex=st.selectbox("Sex",["Female","Male","Other"])
            np_phone=st.text_input("Phone",placeholder="+61 4XX XXX XXX")
            np_email=st.text_input("Email")
        with f3:
            np_med=st.text_input("Medicare",placeholder="XXXX XXXXX X")
            np_ref=st.text_input("Referring Physician")
            np_allrg=st.text_input("Allergies",placeholder="NKDA or list")
            np_emerg=st.text_input("Emergency Contact")
        np_prio=st.selectbox("Priority",["Routine","High","Urgent"])
        np_comp=st.text_area("Chief Complaint *",height=60)
        np_hist=st.text_area("Medical History",height=60)
        sub2=st.form_submit_button("Register Patient",type="primary",use_container_width=True)

    if sub2:
        errs=[]
        if not np_name.strip(): errs.append("Full name required.")
        if not np_pid.strip(): errs.append("Patient ID required.")
        elif any(p.get("patient_id","").upper()==np_pid.strip().upper() for p in patients.values()):
            errs.append(f"Patient ID {np_pid} already exists.")
        if not np_mrn.strip(): errs.append("MRN required.")
        elif any(p.get("mrn","").upper()==np_mrn.strip().upper() for p in patients.values()):
            errs.append(f"MRN {np_mrn} already exists.")
        if not np_comp.strip(): errs.append("Chief complaint required.")
        for e in errs: st.error(e)
        if not errs:
            today=datetime.date.today().strftime("%d %b %Y")
            nid=f"P{len(patients)+1:03d}"
            patients[nid]={
                "id":nid,"patient_id":np_pid.strip(),"name":np_name.strip(),
                "age":int(np_age),"sex":np_sex,"dob":np_dob.strip() or "—",
                "mrn":np_mrn.strip(),"phone":np_phone.strip() or "—",
                "email":np_email.strip() or "—","medicare":np_med.strip() or "—",
                "allergies":np_allrg.strip() or "NKDA","emergency_contact":np_emerg.strip() or "—",
                "priority":np_prio,"last_visit":today,"next_visit":"TBC",
                "registered_date":today,"referring":np_ref.strip() or "—",
                "complaint":np_comp.strip(),"history":np_hist.strip() or "None recorded.",
                "status":"pending","prior_scans":[],"ai_result":None,
                "flagged_to":None,"flag_time":None,"notes":"",
            }
            st.session_state.patients=patients; st.session_state.chat[nid]=[]
            st.session_state.selected_patient=nid; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PROFILE
# ══════════════════════════════════════════════════════════════════════════════
else:
    pid=st.session_state.selected_patient
    if pid not in patients: st.error("Patient not found."); st.stop()
    p=patients[pid]

    is_n=_is_new(p)
    status_label={"urgent":"🚨 URGENT","review":"⚠ FOR REVIEW","clear":"✓ CLEAR",
                  "pending":"○ PENDING ANALYSIS","discharged":"🏥 DISCHARGED"}

    st.markdown(f"""
    <div class='main-header'>
      <div style='display:flex;justify-content:space-between;align-items:center;'>
        <div>
          <h1>👤 {p['name']}</h1>
          <p>{p.get('patient_id','')} · {p['mrn']} · Age {p['age']} · {p['sex']} · DOB: {p['dob']} · {p['referring']}</p>
        </div>
        <div style='background:{status_col.get(p["status"],"#6366f1")};color:white;
                    padding:0.4rem 0.9rem;border-radius:7px;font-weight:700;font-size:0.9rem;'>
          {status_label.get(p["status"],p["status"].upper())}
          {"&nbsp; 🆕" if is_n else ""}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Navigation bar: Back to Dashboard + Logout ────────────────────────────
    st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
    _n1, _n2, _n3 = st.columns([1.1, 1.1, 8])
    with _n1:
        if st.button("← Dashboard", key=f"nav_back_{pid}", use_container_width=True):
            st.session_state.selected_patient = None
            st.session_state.page = 0
            st.rerun()
    with _n2:
        if st.button("🚪 Logout", key=f"nav_logout_{pid}", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    left, mid, right = st.columns([2, 3.2, 1.8])

    # ── LEFT ──────────────────────────────────────────────────────────────────
    with left:
        st.markdown("<div class='sec'>Patient Information</div>",unsafe_allow_html=True)
        al_col="#dc2626" if p.get("allergies","NKDA")!="NKDA" else "#059669"

        for lbl,val,extra in [
            ("Patient ID",p.get("patient_id","—"),""),
            ("MRN",p["mrn"],""),("DOB",p["dob"],""),
            ("Last visit",p["last_visit"],""),("Next visit",p["next_visit"],""),
            ("Phone",p.get("phone","—"),""),("Email",p.get("email","—"),""),
            ("Medicare",p.get("medicare","—"),""),
            ("Emergency",p.get("emergency_contact","—"),""),
            ("Priority",p.get("priority","—"),""),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:0.22rem 0;"
                f"border-bottom:1px solid #f8fafc;font-size:0.82rem;'>"
                f"<span style='color:#64748b;font-weight:500;min-width:80px;'>{lbl}</span>"
                f"<span style='color:#1e293b;text-align:right;'>{val}</span></div>",
                unsafe_allow_html=True)

        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:0.22rem 0;"
            f"font-size:0.82rem;'>"
            f"<span style='color:#64748b;font-weight:500;'>Allergies</span>"
            f"<span style='color:{al_col};font-weight:600;'>{p.get('allergies','NKDA')}</span></div>",
            unsafe_allow_html=True)

        # Editable Patient ID & next visit
        st.markdown("<br>",unsafe_allow_html=True)
        new_pid_val=st.text_input("Edit Patient ID",value=p.get("patient_id",""),key=f"epid_{pid}")
        if new_pid_val.strip() and new_pid_val.strip()!=p.get("patient_id",""):
            conflict=any(op.get("patient_id","")==new_pid_val.strip() and oid!=pid for oid,op in patients.items())
            if conflict: st.error("Patient ID already in use.")
            else: patients[pid]["patient_id"]=new_pid_val.strip()

        try: nv_d=datetime.datetime.strptime(p.get("next_visit","22 May 2026"),"%d %b %Y").date()
        except: nv_d=datetime.date.today()+datetime.timedelta(days=30)
        nv_new=st.date_input("Next Visit",value=nv_d,key=f"nv_{pid}")
        patients[pid]["next_visit"]=nv_new.strftime("%d %b %Y")

        # Status + priority
        s_opts=["urgent","review","pending","clear","discharged"]
        s_idx=s_opts.index(patients[pid]["status"]) if patients[pid]["status"] in s_opts else 0
        new_st=st.selectbox("Status",s_opts,index=s_idx,key=f"st_{pid}")
        p_opts=["Routine","High","Urgent"]
        p_idx=p_opts.index(patients[pid].get("priority","Routine")) if patients[pid].get("priority","Routine") in p_opts else 0
        new_pr=st.selectbox("Priority",p_opts,index=p_idx,key=f"pr_{pid}")
        if st.button("💾 Save Status & Priority",key=f"ssp_{pid}",use_container_width=True):
            patients[pid]["status"]=new_st; patients[pid]["priority"]=new_pr
            st.session_state.patients=patients
            st.success(f"Saved — {new_st.upper()}, {new_pr}"); st.rerun()

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<div class='sec'>Complaint &amp; History</div>",unsafe_allow_html=True)
        st.markdown(f"<div style='background:#f8fafc;padding:0.5rem;border-radius:6px;"
                    f"font-size:0.8rem;color:#374151;margin-bottom:0.35rem;'>{p['complaint']}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div style='background:#f8fafc;padding:0.5rem;border-radius:6px;"
                    f"font-size:0.8rem;color:#374151;'>{p['history']}</div>",
                    unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<div class='sec'>Scan History</div>",unsafe_allow_html=True)
        if p["prior_scans"]:
            for sc in p["prior_scans"]:
                icon=("🔴" if any(w in sc["result"] for w in ["alignant","uspicious"])
                      else "🟢" if any(w in sc["result"] for w in ["on-m","egative","enign","lear","No evidence"])
                      else "⚪")
                st.markdown(f"<div class='scan-item'>{icon} <b>{sc['date']}</b> — {sc['type']}<br>"
                            f"<span style='color:#64748b;'>{sc['result']}</span></div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.8rem;color:#94a3b8;'>No prior records.</div>",
                        unsafe_allow_html=True)

    # ── MIDDLE ─────────────────────────────────────────────────────────────────
    with mid:
        existing = p.get("ai_result") or st.session_state.scan_results.get(pid)
        if existing:
            st.markdown("<div class='sec'>Last AI Analysis</div>",unsafe_allow_html=True)
            pred=existing["class"]; prob_e=existing["probability"]
            css_e="result-cancer" if pred=="Cancer" else "result-clear"
            icon_e="⚠ Malignant Classification" if pred=="Cancer" else "✓ Non-Malignant Classification"
            col_e="#dc2626" if pred=="Cancer" else "#059669"
            st.markdown(f"""
            <div class='{css_e}'>
              <div style='font-size:1.1rem;font-weight:700;color:{col_e};'>{icon_e}</div>
              <div style='font-size:0.8rem;margin-top:0.2rem;'>
                Malignant probability: <b>{prob_e*100:.1f}%</b> · Threshold: 40% · {existing.get('performed','')}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec' style='margin-top:0.75rem;'>AI Scan Analysis — Upload &amp; Run</div>",
                    unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload histopathology scan (.png / .jpg)",
                                     type=["png","jpg","jpeg"], key=f"up_{pid}")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            ic1,ic2 = st.columns(2)
            with ic1: st.image(img, caption="Uploaded scan", use_container_width=True)

            stored_gc = st.session_state.get(f"gc_{pid}")
            if stored_gc is not None:
                with ic2: st.image(stored_gc, caption="Grad-CAM heatmap", use_container_width=True)

            if model:
                if st.button("▶ Run AI Analysis", type="primary", key=f"run_{pid}", use_container_width=True):
                    with st.spinner("Analysing with EfficientNet-B0..."):
                        img_t=_preprocess(img); pred,prob=_inference(model,img_t,device)
                        overlay=_gradcam(model,img_t,device)
                    result={"class":pred,"probability":prob,"performed":datetime.datetime.now().strftime("%d %b %Y %H:%M")}
                    st.session_state.scan_results[pid]=result
                    st.session_state[f"gc_{pid}"]=overlay
                    patients[pid]["ai_result"]=result
                    if pred=="Cancer" and prob>=0.8: patients[pid]["status"]="urgent"
                    elif pred=="Cancer": patients[pid]["status"]="review"
                    # Auto-generate draft clinical note (only if notes empty)
                    draft = _auto_draft_note(pred, prob, p)
                    if not patients[pid].get("notes","").strip():
                        patients[pid]["notes"] = draft
                    # Reset signoff draft so it reflects new AI result
                    st.session_state[f"signoff_draft_{pid}"] = _auto_draft_signoff(pred, prob)
                    # Clear old PDF so download button reflects new analysis
                    st.session_state.pop(f"pdf_{pid}", None)
                    st.session_state.patients=patients
                    st.session_state.chat[pid]=[]
                    st.rerun()
            else:
                st.warning(f"Model not loaded: {model_error}")
        else:
            st.markdown("<div style='font-size:0.8rem;color:#94a3b8;'>Upload a PNG/JPG histopathology image to run analysis.</div>",
                        unsafe_allow_html=True)

        new_r = st.session_state.scan_results.get(pid)
        if new_r:
            pred_n=new_r["class"]; prob_n=new_r["probability"]
            css_n="result-cancer" if pred_n=="Cancer" else "result-clear"
            col_n="#dc2626" if pred_n=="Cancer" else "#059669"
            icon_n="⚠ Malignant" if pred_n=="Cancer" else "✓ Non-Malignant"
            st.markdown(f"""
            <div class='{css_n}' style='margin-top:0.5rem;'>
              <div style='font-size:1.05rem;font-weight:700;color:{col_n};'>{icon_n} — New Analysis</div>
              <div style='font-size:0.8rem;margin-top:0.2rem;'>
                Probability: <b>{prob_n*100:.1f}%</b> · {new_r.get('performed','')}
              </div>
            </div>""", unsafe_allow_html=True)

            m1,m2,m3,m4=st.columns(4)
            for cm_,v_,l_ in [(m1,f"{prob_n*100:.0f}%","Mal. Prob."),
                               (m2,"91.6%","Accuracy"),(m3,"89.3%","Recall"),(m4,"0.970","AUC")]:
                with cm_:
                    st.markdown(f"<div class='metric-box'><div class='val'>{v_}</div><div class='lbl'>{l_}</div></div>",
                                unsafe_allow_html=True)

        active_r = new_r or existing
        st.markdown("<div class='sec' style='margin-top:0.75rem;'>Clinical AI Assistant (Radiology)</div>",
                    unsafe_allow_html=True)

        if not active_r:
            st.markdown("<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:7px;"
                        "padding:0.7rem;font-size:0.8rem;color:#94a3b8;text-align:center;'>"
                        "Upload and run AI analysis to activate the Clinical AI Assistant.</div>",
                        unsafe_allow_html=True)
        else:
            ch = st.session_state.chat.get(pid,[])
            qqs = quick_qs(active_r)
            qc1,qc2,qc3=st.columns(3)
            for qcol,q in zip([qc1,qc2,qc3],qqs):
                with qcol:
                    if st.button(q,key=f"qq_{pid}_{q[:10]}",use_container_width=True):
                        last_q=ch[-2]["content"] if len(ch)>=2 else ""
                        if last_q!=q:
                            with st.spinner("Consulting AI..."):
                                ans=ask_assistant(q,p,active_r,ch)
                            ch.append({"role":"user","content":q})
                            ch.append({"role":"assistant","content":ans})
                            st.session_state.chat[pid]=ch
                        st.rerun()

            # Chat via iframe for proper scroll
            if ch:
                msgs_html=""
                for msg in ch:
                    content=(msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
                    if msg["role"]=="user":
                        msgs_html+=f"<div class='cu'>👨‍⚕️ {content}</div>"
                    else:
                        msgs_html+=f"<div class='cb'>🤖 {content}</div>"
                components.html(f"""
                <style>
                *{{box-sizing:border-box;margin:0;padding:0;}}
                body{{font-family:'Inter',system-ui,sans-serif;background:transparent;}}
                .cw{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:8px;height:240px;overflow-y:auto;
                    display:flex;flex-direction:column;gap:4px;}}
                .cu{{background:#eff6ff;border-radius:10px 10px 3px 10px;
                    padding:6px 10px;max-width:78%;align-self:flex-end;
                    font-size:12.5px;color:#1e293b;word-wrap:break-word;}}
                .cb{{background:white;border:1px solid #e2e8f0;border-radius:10px 10px 10px 3px;
                    padding:6px 10px;max-width:84%;align-self:flex-start;
                    font-size:12.5px;color:#1e293b;word-wrap:break-word;}}
                </style>
                <div class='cw' id='c'>{msgs_html}</div>
                <script>var c=document.getElementById('c');if(c)c.scrollTop=c.scrollHeight;</script>
                """, height=256, scrolling=False)

            with st.form(key=f"cf_{pid}", clear_on_submit=True):
                fi1,fi2=st.columns([5,1])
                with fi1:
                    uq=st.text_input("",placeholder="Ask about this scan (Enter or click Ask Assistant)",
                                     label_visibility="collapsed")
                with fi2:
                    csub=st.form_submit_button("Ask Assistant",use_container_width=True)
                if csub and uq.strip():
                    with st.spinner("Consulting AI..."):
                        ans=ask_assistant(uq,p,active_r,st.session_state.chat.get(pid,[]))
                    st.session_state.chat.setdefault(pid,[])
                    st.session_state.chat[pid]+=[{"role":"user","content":uq},{"role":"assistant","content":ans}]
                    st.rerun()

            st.markdown(
                f"<div class='disclaimer'>⚕ RadiScan is a decision-support tool. "
                f"Requires clinical validation. Not for standalone diagnosis. "
                f"{'GPT-4o active.' if nlp_ok else 'API key not set — template mode.'}</div>",
                unsafe_allow_html=True)

    # ── RIGHT ──────────────────────────────────────────────────────────────────
    with right:
        # Current status
        cur=patients[pid]["status"]
        bg_map={"urgent":"#fef2f2","review":"#fefce8","clear":"#f0fdf4","pending":"#f8fafc","discharged":"#f5f3ff"}
        _sbg = bg_map.get(cur, "#f8fafc")
        _scl = status_col.get(cur, "#6366f1")
        _slb = status_label.get(cur, cur.upper())
        st.markdown(
            f"<div class='status-box' style='background:{_sbg};"
            f"color:{_scl};border-left:3px solid {_scl};'>"
            f"Status: {_slb}</div>",
            unsafe_allow_html=True)

        st.markdown("<div class='sec'>Clinical Actions</div>",unsafe_allow_html=True)

        if st.button("🚨 Flag URGENT — Notify Oncologist",key=f"fu_{pid}",
                     use_container_width=True,type="primary"):
            patients[pid]["status"]="urgent"
            patients[pid]["flagged_to"]="Dr. James Okafor (Oncology)"
            patients[pid]["flag_time"]=datetime.datetime.now().strftime("%d %b %Y %H:%M")
            st.session_state.patients=patients
            st.session_state.notifications.append(
                f"URGENT: {p['name']} → Oncology ({patients[pid]['flag_time']})")
            st.rerun()

        if st.button("⚠ Flag for MDT Review",key=f"fr_{pid}",use_container_width=True):
            patients[pid]["status"]="review"; patients[pid]["flagged_to"]=None
            patients[pid]["flag_time"]=None; st.session_state.patients=patients; st.rerun()

        if st.button("✓ Mark as Clear",key=f"fc_{pid}",use_container_width=True):
            patients[pid]["status"]="clear"; patients[pid]["flagged_to"]=None
            patients[pid]["flag_time"]=None; st.session_state.patients=patients; st.rerun()

        if st.button("🏥 Mark as Discharged",key=f"fd_{pid}",use_container_width=True):
            patients[pid]["status"]="discharged"; st.session_state.patients=patients; st.rerun()

        if patients[pid].get("flagged_to"):
            st.markdown(
                f"<div class='flag-box'><b>📬 Flagged to:</b><br>"
                f"{patients[pid]['flagged_to']}<br>"
                f"<span style='color:#6b7280;font-size:0.75rem;'>{patients[pid].get('flag_time','')}</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(
            "<div class='sec'>Clinical Notes"
            "<span style='font-weight:400;font-size:0.65rem;color:#94a3b8;"
            "margin-left:0.5rem;letter-spacing:0;text-transform:none;'>"
            "AI draft generated after analysis</span></div>",
            unsafe_allow_html=True)

        cur_notes=patients[pid].get("notes","")
        if cur_notes:
            st.markdown(f"<div class='notes-box'><b>📋 Current:</b><br>{cur_notes}</div>",
                        unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)

        notes = st.text_area("", value=cur_notes, height=110, key=f"n_{pid}",
                             placeholder=(
                                 "Run AI Analysis to auto-generate a draft note. "
                                 "Edit, refine, or replace before saving."
                                 if not cur_notes else "Edit clinical notes..."
                             ),
                             label_visibility="collapsed")
        if st.button("💾 Save Notes",key=f"sv_{pid}",use_container_width=True):
            patients[pid]["notes"]=notes; st.session_state.patients=patients
            st.success("Saved."); st.rerun()

        # PDF
        st.markdown("<br>",unsafe_allow_html=True)
        # Define active_for_pdf FIRST — used everywhere in this section
        active_for_pdf = st.session_state.scan_results.get(pid) or patients[pid].get("ai_result")

        st.markdown(
            "<div class='sec'>Patient Report"
            "<span style='font-weight:400;font-size:0.65rem;color:#94a3b8;"
            "margin-left:0.5rem;letter-spacing:0;text-transform:none;'>"
            + ("Sign-off auto-drafted · edit before generating" if active_for_pdf
               else "Run AI Analysis first") +
            "</span></div>",
            unsafe_allow_html=True)

        _signoff_key = f"rc_{pid}"
        # Only auto-populate sign-off if not yet edited by user this session
        _signoff_default = st.session_state.get(f"signoff_draft_{pid}", None)
        if _signoff_default is None and active_for_pdf:
            _signoff_default = _auto_draft_signoff(
                active_for_pdf["class"], active_for_pdf["probability"])
            st.session_state[f"signoff_draft_{pid}"] = _signoff_default
        elif _signoff_default is None:
            _signoff_default = ""

        rad_comment = st.text_area("", height=60, key=_signoff_key,
                                   value=_signoff_default,
                                   placeholder="Radiologist sign-off comment (optional)...",
                                   label_visibility="collapsed")

        if st.button("📄 Generate & Preview Report", key=f"gpdf_{pid}", use_container_width=True):
            if not active_for_pdf:
                st.warning("No AI analysis available. Run AI Analysis first to generate a report.")
            else:
                try:
                    final_notes = (notes + ("\n\nRadiologist Sign-off: " + rad_comment
                                   if rad_comment.strip() else "")).strip()
                    pdf_b = _generate_pdf(patients[pid], active_for_pdf, final_notes, rad_comment)
                    st.session_state[f"pdf_{pid}"] = pdf_b
                    pred_  = active_for_pdf["class"]
                    prob_  = f"{active_for_pdf['probability']*100:.1f}%"
                    pcol_  = "#dc2626" if pred_ == "Cancer" else "#059669"
                    ts_    = active_for_pdf.get("performed", "—")
                    flagged_ = patients[pid].get("flagged_to") or "—"
                    st.markdown(
                        f"<div style='background:white;border:1px solid #e2e8f0;border-radius:7px;"
                        f"padding:0.75rem;font-size:0.78rem;margin-top:0.4rem;line-height:1.7;'>"
                        f"<div style='font-weight:700;color:#0f172a;margin-bottom:0.35rem;'>"
                        f"📋 Report Preview — {patients[pid]['name']}</div>"
                        f"<b>AI Result:</b> "
                        f"<span style='color:{pcol_};font-weight:700;'>{pred_} ({prob_})</span>"
                        f" · {ts_}<br>"
                        f"<b>Patient ID:</b> {patients[pid].get('patient_id','—')}"
                        f" · <b>MRN:</b> {patients[pid]['mrn']}<br>"
                        f"<b>Priority:</b> {patients[pid].get('priority','—')}"
                        f" · <b>Allergies:</b> {patients[pid].get('allergies','—')}<br>"
                        f"<b>Referred to:</b> {flagged_}<br>"
                        f"<b>Sign-off:</b> {rad_comment[:80] + '...' if len(rad_comment) > 80 else rad_comment or '—'}"
                        f"</div>", unsafe_allow_html=True)
                    st.success("PDF generated — click Download below.")
                except Exception as e:
                    st.error(f"PDF error: {e}")

        _pdf_data = st.session_state.get(f"pdf_{pid}")
        if _pdf_data:
            st.download_button(
                "⬇ Download PDF Report",
                data=_pdf_data,
                file_name=(f"RadiScan_{patients[pid].get('patient_id', patients[pid]['mrn'])}"
                           f"_{datetime.date.today().strftime('%Y%m%d')}.pdf"),
                mime="application/pdf",
                key=f"dl_{pid}_{len(_pdf_data)}",
                use_container_width=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("<div class='sec'>System</div>",unsafe_allow_html=True)
        st.markdown(f"EfficientNet-B0 v0.2  |  AWS SageMaker")
        st.markdown(f"GPU: {'Tesla T4 ✅' if torch.cuda.is_available() else 'CPU ⚠'}")
        st.markdown(f"NLP: GPT-4o {'✅' if nlp_ok else '⚠ no key'}")
