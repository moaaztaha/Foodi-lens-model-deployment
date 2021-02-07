from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision.all import *
import math

model_file_url = 'https://github.com/JoannaDiao/FoodieLens/blob/main/web-app/app/models/export.pkl?raw=true'
model_file_name = 'export.pkl'
path = Path(__file__).parent

templates = Jinja2Templates(directory='app/templates')

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

api_key = 'AIzaSyBOp0pH8QYUOc1E0CbHU8a9_N2Dk0JmJBU'
# url variable store url 
url = "https://maps.googleapis.com/maps/api/place/textsearch/json?"

dict = {}
file = open("message.txt")
for line in file:
  key, value = line.split()
  dict[int(key)] = value


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}')
    try:
        learn = load_learner(path/'models'/model_file_name, cpu=True)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

@app.route("/upload", methods=["POST"])
async def upload(request):
    form = await request.form()
    img_bytes = await (form["file"].read())
    output, cords = predict_from_bytes(img_bytes)
    return templates.TemplateResponse('result.html', {'request': request, 'output': output, 'cords': cords})


def predict_from_bytes(img_bytes):
    pred,pred_idx,probs = learn.predict(img_bytes)
    classes = learn.dls.vocab
    predictions = sorted(zip(classes, map(float, probs)), key=lambda p: p[1], reverse=True)
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    
    output = ''
    for pred in predictions[0:3]:
        output+= dict[pred[0]] +  ' ' + str(float("{:.3f}".format(pred[1]))) + ' |\t'
    
    print(dict[predictions[0][0]])
    search_output = search(dict[predictions[0][0]])
        


    result_html = str(result_html1.open().read() + output + result_html2.open().read())
    # return HTMLResponse(result_html)
    return output, search_output

def search(prediction):
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={prediction}&inputtype=textquery&fields=formatted_address,name,rating,opening_hours,geometry&key=AIzaSyBOp0pH8QYUOc1E0CbHU8a9_N2Dk0JmJBU"
    #url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?query={prediction}&inputtype=textquery&fields=formatted_address,name,rating,opening_hours,geometry&key=AIzaSyBOp0pH8QYUOc1E0CbHU8a9_N2Dk0JmJBU"

    r = requests.get(url)
    x = r.json()
    print(x)
    if x != None:
        lat, lng = x['candidates'][0]['geometry']['location']['lat'], x['candidates'][0]['geometry']['location']['lng']
        return { 'lat': lat, 'lng': lng }
    else:
        return 31.1342, 29.9792
    
    


@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
    #uvicorn.run(app, host='0.0.0.0', port=8000)