from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
import base64
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://onedrive.live.com/download?cid=B64B3F2B0AD5009F&resid=B64B3F2B0AD5009F%211234&authkey=AJ7x7MzKjgEiysA'
export_file_name = 'cloudsocsingleunfrozen.pkl'

classes = ['Altocumulus', 'Altostratus', 'Cirrocumulus', 'Cirrostratus', 'Cirrus', 'Cumulonimbus', 'Cumulus', 'Nimbostratus', 'NotACloud', 'Stratocumulus', 'Stratus']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
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

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    apiKey = data['apiKey']
    if apiKey !='7a966000-a16d-48eb-8186-d820fea2e48a':
        return JSONResponse({'result': 'Wrong API Key'})
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img,thresh=0.2)[0]
    return JSONResponse({'result': str(prediction)})
	
	
@app.route('/analyzeAPI', methods=['POST'])
async def analyzeAPI(request):
    data = await request.form()
    apiKey = data['apiKey']
    if apiKey !='7a966000-a16d-48eb-8186-d820fff2e48a':
        return JSONResponse({'result': 'Wrong API Key'})
    img_bytes = base64.b64decode(data['fileBase64'])
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img,thresh=0.2)[0]
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
