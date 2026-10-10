import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

load_dotenv()
DATA = Path(__file__).parent / 'data' / 'sample_air_quality.csv'
AIR_URL = 'https://air-quality-api.open-meteo.com/v1/air-quality'
WEATHER_URL = 'https://api.open-meteo.com/v1/forecast'
LAGS = [1, 2, 3, 6, 12, 24, 48]
WEATHER = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
FEATURES = [f'lag_{n}' for n in LAGS] + ['mean6', 'mean24', 'hour', 'dayofweek'] + WEATHER
st.set_page_config(page_title='AirAware AI', page_icon='🌫️', layout='wide')
st.title('🌫️ AirAware AI')
st.subheader('AI-powered air quality forecaster & health advisor')
st.caption('Educational prototype. Forecasts are estimates, not official monitoring readings or medical advice.')

def sample_data():
    if DATA.exists():
        return pd.read_csv(DATA, parse_dates=['time'])
    rng=np.random.default_rng(17); t=pd.date_range(end=pd.Timestamp.now().floor('h'), periods=24*45, freq='h')
    h=t.hour.to_numpy(); d=np.arange(len(t)); pm=65+24*np.sin(2*np.pi*(h-7)/24)+18*np.sin(2*np.pi*d/(24*8))+rng.normal(0,9,len(t))
    return pd.DataFrame({'time':t,'pm2_5':np.clip(pm,5,220),'temperature_2m':24+7*np.sin(2*np.pi*(h-6)/24)+rng.normal(0,1.5,len(t)),'relative_humidity_2m':np.clip(55-15*np.sin(2*np.pi*(h-6)/24)+rng.normal(0,5,len(t)),15,95),'wind_speed_10m':np.clip(7+rng.normal(0,3,len(t)),0,25)})

def fetch(lat, lon):
    def get(url, params):
        r=requests.get(url,params=params,timeout=25); r.raise_for_status(); return r.json()['hourly']
    a=get(AIR_URL, {'latitude':lat,'longitude':lon,'hourly':'pm2_5,pm10,us_aqi,european_aqi','past_days':30,'forecast_days':3,'timezone':'auto'})
    w=get(WEATHER_URL, {'latitude':lat,'longitude':lon,'hourly':'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation','past_days':30,'forecast_days':3,'timezone':'auto'})
    ad=pd.DataFrame(a); wd=pd.DataFrame(w); ad['time']=pd.to_datetime(ad.time); wd['time']=pd.to_datetime(wd.time)
    df=ad.merge(wd,on='time',how='inner').sort_values('time').drop_duplicates('time').dropna(subset=['pm2_5']).reset_index(drop=True)
    if len(df)<24*7: raise ValueError('Not enough hourly data returned by API.')
    return df

def features(df):
    x=df.copy().sort_values('time').reset_index(drop=True)
    for n in LAGS: x[f'lag_{n}']=x.pm2_5.shift(n)
    x['mean6']=x.pm2_5.shift(1).rolling(6).mean(); x['mean24']=x.pm2_5.shift(1).rolling(24).mean()
    x['hour']=x.time.dt.hour; x['dayofweek']=x.time.dt.dayofweek
    for c in WEATHER:
        if c not in x: x[c]=0.0
    return x

def aqi(pm):
    c=max(0,np.floor(float(pm)*10)/10)
    bands=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for lo,hi,il,ih in bands:
        if lo<=c<=hi:return int(round((ih-il)/(hi-lo)*(c-lo)+il))
    return 500

def category(n):
    if n<=50:return 'Good'
    if n<=100:return 'Moderate'
    if n<=150:return 'Unhealthy for sensitive groups'
    if n<=200:return 'Unhealthy'
    if n<=300:return 'Very unhealthy'
    return 'Hazardous'

def advise(n, peak_hour, city):
    tips=[]
    if n<=50: tips=['Air quality is relatively good; usual outdoor activities are generally reasonable.']
    elif n<=100: tips=['Sensitive people may reduce prolonged or intense outdoor exertion if symptoms occur.']
    elif n<=150: tips=['Children, older adults, and people with heart or lung disease should reduce prolonged outdoor exertion.','Consider moving strenuous exercise indoors.']
    else: tips=['Reduce strenuous outdoor activity and minimize time outside where possible.','Keep windows closed during pollution peaks and use a suitable indoor particle air cleaner if available.','If you must go out, a well-fitting N95/FFP2 respirator can reduce particle exposure.']
    tips.append(f'Modelled peak is around {peak_hour:02d}:00 local time; consider moving outdoor activity away from that period.')
    tips.append('Follow official local air-quality alerts. Seek urgent medical help for severe breathlessness or chest pain.')
    key=os.getenv('OPENAI_API_KEY','').strip()
    if key:
        try:
            from openai import OpenAI
            client=OpenAI(api_key=key)
            res=client.responses.create(model=os.getenv('OPENAI_MODEL','gpt-4.1-mini'),input=f'Give 3-5 concise cautious health tips for {city}. Estimated PM2.5-based US AQI: {n}. Peak hour: {peak_hour}. Distinguish sensitive groups, do not diagnose, mention official local alerts.')
            return 'Optional LLM guidance', [s.strip(' •-') for s in res.output_text.splitlines() if s.strip()]
        except Exception: pass
    return 'Built-in health advisor', tips

with st.sidebar:
    st.header('Settings'); city=st.text_input('Location label','New Delhi, India')
    lat=st.number_input('Latitude',value=28.6139,min_value=-90.0,max_value=90.0,format='%.4f')
    lon=st.number_input('Longitude',value=77.2090,min_value=-180.0,max_value=180.0,format='%.4f')
    horizon=st.select_slider('Forecast horizon',options=[24,48],value=48,format_func=lambda x:f'{x} hours')
    demo=st.toggle('Use included demo dataset',value=False)
    run=st.button('Generate forecast',type='primary')
if 'go' not in st.session_state: st.session_state.go=True
if run: st.session_state.go=True
if st.session_state.go:
  try:
    if demo:
      df=sample_data(); hist=df.copy(); now=hist.time.max(); future_t=pd.date_range(now+pd.Timedelta(hours=1),periods=horizon,freq='h'); last=hist.iloc[-1]
      fw=pd.DataFrame({'time':future_t,'temperature_2m':[float(last.get('temperature_2m',25))+4*np.sin(2*np.pi*t.hour/24) for t in future_t],'relative_humidity_2m':[float(last.get('relative_humidity_2m',50))]*horizon,'wind_speed_10m':[float(last.get('wind_speed_10m',7))]*horizon})
      st.warning('Demo dataset is illustrative synthetic data, not live measurements.')
    else:
      with st.spinner('Fetching public air-quality and weather data…'): df=fetch(lat,lon)
      now=pd.Timestamp.now().floor('h'); hist=df[df.time<=now].copy(); fw=df[df.time>now].copy()
      if len(hist)<200: hist=df.dropna(subset=['pm2_5']).iloc[:-48].copy()
      if len(fw)<horizon: raise ValueError(f'Only {len(fw)} future hours available. Try a shorter horizon or retry.')
    hist=hist.dropna(subset=['pm2_5']).sort_values('time').reset_index(drop=True)
    f=features(hist).dropna(subset=FEATURES+['pm2_5'])
    if len(f)<100: raise ValueError('Not enough clean history for training.')
    split=min(max(int(len(f)*.8),48),len(f)-24); tr,te=f.iloc[:split],f.iloc[split:]
    model=RandomForestRegressor(n_estimators=180,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1); model.fit(tr[FEATURES],tr.pm2_5)
    pred=np.clip(model.predict(te[FEATURES]),0,None); base=te['lag_1'].to_numpy(); actual=te.pm2_5.to_numpy()
    mae=mean_absolute_error(actual,pred); bmae=mean_absolute_error(actual,base); rmse=np.sqrt(mean_squared_error(actual,pred))
    model.fit(f[FEATURES],f.pm2_5)
    hist2=hist[['time','pm2_5']+[c for c in WEATHER if c in hist]].copy(); out=[]
    for _,r in fw.sort_values('time').drop_duplicates('time').head(horizon).iterrows():
      t=r.time; row={f'lag_{n}':float(hist2.pm2_5.iloc[-n]) for n in LAGS}; row.update({'mean6':float(hist2.pm2_5.tail(6).mean()),'mean24':float(hist2.pm2_5.tail(24).mean()),'hour':t.hour,'dayofweek':t.dayofweek})
      for c in WEATHER: row[c]=float(r.get(c,0)) if pd.notna(r.get(c,0)) else 0.0
      pm=float(np.clip(model.predict(pd.DataFrame([row])[FEATURES])[0],0,500)); out.append({'time':t,'predicted_pm2_5':pm,'estimated_us_aqi':aqi(pm),'temperature_2m':row['temperature_2m'],'relative_humidity_2m':row['relative_humidity_2m'],'wind_speed_10m':row['wind_speed_10m']})
      hist2=pd.concat([hist2,pd.DataFrame([{'time':t,'pm2_5':pm,**{c:row[c] for c in WEATHER}}])],ignore_index=True)
    fc=pd.DataFrame(out)
    if fc.empty: raise ValueError('No forecast rows were generated.')
    peak=fc.loc[fc.predicted_pm2_5.idxmax()]; latest=float(hist.pm2_5.iloc[-1]); cur=aqi(latest); peak_aqi=int(fc.estimated_us_aqi.max())
    st.caption(f'{city} · Times follow the data provider timezone. '+('Demo mode' if demo else 'Live public API data; may differ from ground sensors.'))
    a,b,c,d=st.columns(4); a.metric('Latest PM2.5',f'{latest:.1f} µg/m³'); b.metric('Approx. US AQI',f'{cur}'); c.metric(f'Peak PM2.5 ({horizon}h)',f'{peak.predicted_pm2_5:.1f} µg/m³'); d.metric('Peak category',category(peak_aqi))
    st.markdown('### PM2.5 forecast'); st.line_chart(fc.set_index('time')[['predicted_pm2_5']].rename(columns={'predicted_pm2_5':'Predicted PM2.5 (µg/m³)'}),use_container_width=True)
    st.markdown('### Forecast quality check'); x,y,z=st.columns(3); x.metric('Model MAE',f'{mae:.2f} µg/m³'); y.metric('Persistence baseline MAE',f'{bmae:.2f} µg/m³'); z.metric('Model RMSE',f'{rmse:.2f} µg/m³')
    if mae<bmae: st.success('Model beat the persistence baseline on this holdout split.')
    else: st.info('Model did not beat the baseline on this split. Report honestly and tune with more data.')
    st.caption(f'Chronological holdout: {len(te)} hours. AQI is an approximate US PM2.5-only estimate, not CPCB AQI.')
    st.markdown('### Health advisor'); title,tips=advise(peak_aqi,pd.Timestamp(peak.time).hour,city); st.write('**'+title+'**')
    for tip in tips: st.markdown('- '+tip)
    st.caption('General education only, not medical advice. Follow official alerts and your clinician.')
    st.markdown('### Hourly forecast data'); st.dataframe(fc,use_container_width=True,hide_index=True)
    st.download_button('Download forecast CSV',fc.to_csv(index=False).encode(),file_name='air_quality_forecast.csv',mime='text/csv')
    st.download_button('Download training data CSV',hist.to_csv(index=False).encode(),file_name='air_quality_training_data.csv',mime='text/csv')
  except requests.RequestException as e:
    st.error('Could not reach the public API. Check your internet connection or enable demo mode.'); st.caption(type(e).__name__)
  except Exception as e: st.error(f'Could not generate forecast: {e}')
with st.expander('How it works and limitations'):
 st.markdown('''1. Fetch hourly PM2.5 and weather data from Open-Meteo.\n2. Create lag, rolling-average, time and weather features.\n3. Evaluate Random Forest against a previous-hour persistence baseline using a chronological holdout.\n4. Forecast recursively for 24/48 hours. Errors can accumulate over the forecast horizon.\n5. Offer health tips through built-in rules, with optional OpenAI wording via `OPENAI_API_KEY`.\n\nThis is a portfolio prototype, not a certified pollution-monitoring or medical product. Demo data is synthetic. The AQI conversion is approximate US PM2.5-only, not India's CPCB multi-pollutant AQI.''')
