import os, requests, time
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")  # if you have

def get_access_token():
    url = "https://www.strava.com/oauth/token"
    print(f"client_id {CLIENT_ID} refresh_token {REFRESH_TOKEN}")
    resp = requests.post(url, data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN
    })
    resp.raise_for_status()
    return resp.json()["access_token"]

def fetch_activities(access_token, per_page=200):
    headers = {"Authorization": f"Bearer {access_token}"}
    page = 1
    items=[]
    while (True and page < 10):
        r = requests.get("https://www.strava.com/api/v3/athlete/activities",
                         headers=headers, params={"per_page": per_page, "page": page})
        r.raise_for_status()
        data = r.json()
        if not data: break
        items.extend(data); page += 1
        time.sleep(0.2)
    return items

if __name__ == "__main__":
    token = get_access_token()
    acts = fetch_activities(token)
    print(f"Fetched {len(acts)} activities")
    
    for activity in acts:
        print(activity)
    
