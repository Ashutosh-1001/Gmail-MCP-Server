import os, base64, io, re, pytz, nltk
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
from sse import create_sse_server
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import pipeline

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
IST = pytz.timezone("Asia/Kolkata")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
mcp = FastMCP("gmail-analytics")
app.mount("/mcp", create_sse_server(mcp))


def gmail():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)
        open("token.json", "w").write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def header(h, k): return next((x["value"] for x in h if x["name"].lower()==k.lower()), "")
def parse_date(s):
    for f in ["%a, %d %b %Y %H:%M:%S %z","%a, %d %b %Y %H:%M:%S"]:
        try:
            d = datetime.strptime(s.strip(), f)
            return (d if d.tzinfo else d.replace(tzinfo=pytz.UTC)).astimezone(IST)
        except: pass


def fetch_msgs(q=None, n=10):
    svc = gmail()
    res = svc.users().messages().list(userId="me", q=q, maxResults=n).execute()
    return res.get("messages", []), svc


@mcp.tool()
def check_recent_emails(n: int = 5):
    msgs, svc = fetch_msgs(n=n)
    out = []
    for m in msgs:
        d = svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                      metadataHeaders=["From","Subject","Date"]).execute()
        h = d["payload"]["headers"]
        out.append(f"{header(h,'From')} | {header(h,'Subject')} | {header(h,'Date')}")
    return "\n".join(out) or "No emails"


@mcp.tool()
def list_unread(n: int = 10):
    msgs, svc = fetch_msgs("is:unread", n)
    return "\n".join(
        f"{header(d['payload']['headers'],'From')} → {header(d['payload']['headers'],'Subject')}"
        for m in msgs
        for d in [svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                             metadataHeaders=["From","Subject"]).execute()]
    ) or "Empty"


@mcp.tool()
def send_reply(mid: str, text: str):
    svc = gmail()
    o = svc.users().messages().get(userId="me", id=mid, format="metadata",
                                  metadataHeaders=["From","Subject","Message-ID"]).execute()
    h = o["payload"]["headers"]
    msg = MIMEMultipart()
    msg["To"], msg["Subject"] = header(h,"From"), "Re: "+header(h,"Subject")
    msg.attach(MIMEText(text))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return "Sent"


@mcp.tool()
def delete(mid: str):
    gmail().users().messages().delete(userId="me", id=mid).execute()
    return "Deleted"


@mcp.tool()
def top_senders(n: int = 5):
    msgs, svc = fetch_msgs(n=100)
    c = defaultdict(int)
    for m in msgs:
        d = svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                      metadataHeaders=["From"]).execute()
        c[header(d["payload"]["headers"],"From")] += 1
    return "\n".join(f"{k}: {v}" for k,v in sorted(c.items(), key=lambda x:-x[1])[:n])


@mcp.tool()
def volume():
    msgs, svc = fetch_msgs(n=100)
    b = defaultdict(int)
    for m in msgs:
        d = svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                      metadataHeaders=["Date"]).execute()
        dt = parse_date(header(d["payload"]["headers"],"Date"))
        if dt: b[dt.date()] += 1
    return "\n".join(f"{k}: {v}" for k,v in sorted(b.items()))


@mcp.tool()
def sentiment(n: int = 5):
    clf = pipeline("sentiment-analysis")
    msgs, svc = fetch_msgs(n=n)
    out = []
    for m in msgs:
        d = svc.users().messages().get(userId="me", id=m["id"]).execute()
        txt = d.get("snippet","")
        if txt:
            r = clf(txt[:512])[0]
            out.append(f"{r['label']} {r['score']:.2f} → {txt[:60]}")
    return "\n".join(out)


@mcp.tool()
def topics(n: int = 20):
    msgs, svc = fetch_msgs(n=n)
    freq = defaultdict(int)
    for m in msgs:
        d = svc.users().messages().get(userId="me", id=m["id"]).execute()
        for w in word_tokenize(d.get("snippet","").lower()):
            if w.isalpha() and w not in STOP_WORDS and len(w)>3:
                freq[w]+=1
    return "\n".join(f"{w}: {c}" for w,c in Counter(freq).most_common(10))


@app.get("/")
def health():
    return {"status": "ok"}
