import requests

#airports
request = requests.get('http://api.aviationstack.com/v1/airports?access_key=636cee5fb0b4a0b713e5c16e2f595e3e')
data = request.json()
#for p in data['people']:print(p['name'])
print(data)

#airlines
request = requests.get('http://api.aviationstack.com/v1/airlines?access_key=636cee5fb0b4a0b713e5c16e2f595e3e')
data = request.json()
for p in data['data']:
    print(p['airline_name'] + ' - ' + p['icao_code'])





