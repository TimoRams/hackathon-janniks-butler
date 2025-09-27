from openai import OpenAI
api_key = "xxx"
clientGPT = OpenAI(api_key=api_key)

def chat_with_gpt(prompt, inhalt):
    response = clientGPT.responses.create(
        model="gpt-3.5-turbo", # Specify the model
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": inhalt}
        ]
    )
    return response.output_text



from paho.mqtt import client as mqtt_client

broker = '172.16.1.186'
port = 1883
topic = [("notes/textnachrichten/text", 0), ("notes/textnachrichten/gesicht", 0)]
client_id = f'python-mqtt-luca'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        print("Hallo")
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    print("B")
    client.connect(broker, port)
    print("A")
    return client

def publish(client, topic, msg):
    result = client.publish(topic, msg)
    if result[0] == 0:
        print(f"Sent `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

letztes_gesicht = ""

def on_message(client, userdata, msg):
    global letztes_gesicht
    print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    if msg.topic == "notes/textnachrichten/gesicht":
        letztes_gesicht = msg.payload.decode()
    if msg.topic == "notes/textnachrichten/text":
        nachrichtentext = msg.payload.decode() 
        prompt = "Fasse den dir gegebenen Text auf die Wichtigsten stichpunkte zusammen. am besten in einen kurzen To-Do. Streich aspekte ohne inhalt.Gib bei den aussagen an, wer sie getätigt hat"
        zusammenfassung = chat_with_gpt(prompt, letztes_gesicht + " sagt: " + nachrichtentext)
        print(f"Zusammenfassung: {zusammenfassung}")
        publish(client, "notes/zusammenfasung", zusammenfassung)


def subscribe(client, topic):
    client.subscribe(topic)
    client.on_message = on_message


# Verbindung herstellen und in `client` speichern
client = connect_mqtt()
# Mit dem Topic `*`
subscribe(client, topic)
# Client anweisen, regelmäßig auf Nachrichten zu hören
#client.loop_start()
client.loop_forever()
# Veröffentliche eine Nachricht
