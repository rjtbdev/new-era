from dotenv import load_dotenv
import pathlib, os, traceback
load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parents[1] / ".env")
from circleguard import Circleguard, ReplayPath

cg = Circleguard(os.environ['OSU_API_V1_KEY'])
r = ReplayPath('data/generated/replays/954692_model_default.osr')
try:
    cg.load(r)
    print("LOADED ok. mods=", r.mods, " ur=", cg.ur(r))
except Exception as e:
    traceback.print_exc()
