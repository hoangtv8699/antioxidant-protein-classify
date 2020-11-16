import paho.mqtt.publish as publish
from uuid import getnode as getmac
import paho.mqtt.client as paho
import json
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import serial
import time



def readData():
    while True:
        try:
            ser = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=9600,
            )
            read_serial = ser.readline()
            return read_serial
        except Exception as e:
            return None
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
                client.subscribe("test12", qos=1)

                id = getmac()
                message = {
                    "type": 1,
                    "data": {
                        "id": id
                    }
                }
                publish.single("test12", json.dumps(message), hostname="broker.hivemq.com", port=1883,
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
        data = readData()
        rs = enc.encrypt(data)

        msg = {
            "type": 2,
            "data": base64.encodebytes(rs).decode('utf-8')
        }

        publish.single("test12", json.dumps(msg), hostname="broker.hivemq.com", port=1883,
                       retain=False, qos=1)
        print('published message: ' + str(msg))
        time.sleep(60)
