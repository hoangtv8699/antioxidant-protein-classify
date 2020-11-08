from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64


pk = '''-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuNXH6TU5aH4fIY6gr9fx
rVgg0VHu7uZjYIDnwY4LE7nZwMkAj9XgeEXydGex1WJg69x2cCjS9UKBaa/rTMxy
Doh3NAXyYg3TgX94lRmvoPyIKEIED4P4QE1sBh4SvbouO4SWyzQ9Bc5yLEdL57gA
iGyH9GJT0WfFLE6ki8JULEFXBIDrpPu6sNA1BhhFgGosgHXiw3ghn7YzfQG+RyBv
6tTGFilVXWEgpFlVdIG+XbzkLOgEZT8XdYC3GEubIf/GpFIju98OfjcpY4DIlNqx
gDnWVLHV5I57DZBsv80hePdewjez0sul1+SRGtkrz0ZLhbHJHmzKWZKbuGEhPCnq
NwIDAQAB
-----END PUBLIC KEY-----'''

pkk = RSA.importKey(pk)

enc = PKCS1_OAEP.new(pkk)
msg = b'Hello World'

rs = enc.encrypt(msg)

print(base64.encodebytes(rs))