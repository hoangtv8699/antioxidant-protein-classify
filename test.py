import paho.mqtt.publish as publish
from uuid import getnode as getmac
import paho.mqtt.client as paho
import json
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import serial
import time

ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=9600
)

data = {
    "temp": 0,
    "humidity": 0,
    "light": 54612,
    "ec": 5,
    "ph": 8,
    "waterTemp": 0
}

state = {
    "relay1": 0,
    "relay2": 0,
    "relay3": 0,
    "relay4": 0
}


def readData():
    while True:
        try:
            read_serial = ser.readline()
            jsonTmp = json.loads(read_serial)

            data['temp'] = jsonTmp['temp']
            data['humidity'] = jsonTmp['humidity']
            data['light'] = jsonTmp['light']
            data['ec'] = jsonTmp['ec']
            data['ph'] = jsonTmp['ph']
            data['waterTemp'] = jsonTmp['waterTemp']

            state['relay1'] = jsonTmp['relay1']
            state['relay2'] = jsonTmp['relay2']
            state['relay3'] = jsonTmp['relay3']
            state['relay4'] = jsonTmp['relay4']
        except Exception as e:
            print(e)
            return None
        break


def writeData(data):
    while True:
        try:
            ser.write(data)
            print("write data to serial: " + str(data))
        except Exception as e:
            print(e)
        break


def on_message(client, userdata, message):
    print("Message received: " + str(message.payload))
    try:
        with open('pubkey.txt', 'w+') as f:
            payload = json.loads(message.payload)
            if "pubkey" in payload["data"]:
                f.write(str(payload["data"]["pubkey"]))
                client.loop_stop()
    except:
        print("json parse exception")


def getPubkey():
    try:
        with open('pubkey.txt', 'r') as f:
            pubkey = f.read()
            if pubkey:
                return pubkey
            else:
                client = paho.Client()
                client.on_message = on_message
                client.connect("broker.hivemq.com", 1883)

                client.loop_start()
                client.subscribe("Farm", qos=1)

                id = str(getmac())
                message = {
                    "type": 1,
                    "data": {
                        "id": id
                    }
                }
                publish.single("Farm", json.dumps(message), hostname="broker.hivemq.com", port=1883,
                               retain=False, qos=1)
                while True:
                    pubkey = f.read()
                    if pubkey:
                        return pubkey
    except:
        print("chi biet la co loi")


if __name__ == '__main__':
    pk = getPubkey()
    pkk = RSA.importKey(pk)
    enc = PKCS1_OAEP.new(pkk)
    while True:
        a = {
            "type": 1,
            "data": {
                "relay1": 1,
                "relay2": 1,
                "relay3": 1,
                "relay4": 1
            }
        }
        writeData(json.dumps(a).encode('utf-8'))

        readData()
        print(data)
        rs = enc.encrypt(json.dumps(data).encode('utf-8'))

        msg = {
            "type": 2,
            "data": {
                "id": str(getmac()),
                "data": base64.encodebytes(rs).decode('utf-8')
            }
        }

        publish.single("Farm", json.dumps(msg), hostname="broker.hivemq.com", port=1883,
                       retain=False, qos=1)
        print('published message: ' + str(msg))
        time.sleep(60)
