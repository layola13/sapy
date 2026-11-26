import urllib.parse
import urllib.request

print(urllib.parse.quote("a b/c?d"))
print(urllib.parse.quote_plus("a b+c"))
print(urllib.parse.unquote("a%20b%2Bc"))
print(urllib.parse.unquote_plus("a+b%2Bc"))
print(urllib.parse.urlencode({"name": "a b", "n": "3"}))
print(urllib.parse.urljoin("http://ex.com/a/b/c", "../d"))
print(urllib.parse.urljoin("http://ex.com/a/b/c", "/x"))
print(urllib.parse.parse_qs("a=1&b=&c&a=2"))
req = urllib.request.Request("http://127.0.0.1:18713/hello")
resp = urllib.request.urlopen("http://127.0.0.1:18713/hello")
print("done")
