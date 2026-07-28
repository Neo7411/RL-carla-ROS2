# Dokumentáció — RL-carla-ROS2

Részletes, magyar nyelvű dokumentáció a projekthez. Azért készült, hogy a kód **minden
része érthető** legyen, és biztosan tudj rajta továbbfejleszteni.

## Tartalom

| # | Fájl | Miről szól |
|---|------|-----------|
| 00 | [Áttekintés](00-attekintes.md) | Mi ez a projekt, hogyan épül fel, mi hol van, futási folyamat |
| 01 | [train.py, config.py, utils.py](01-train-config.md) | Indítás, hiperparaméterek, callbackek, LR ütemező |
| 02 | [carla_route_env.py](02-carla-route-env.md) | ★ A Gym környezet — `reset`, `step`, `render`, sorról sorra |
| 03 | [wrappers, rewards, state_commons](03-wrappers-rewards-state.md) | CARLA burkolók, a jutalomfüggvény, az állapot előállítása |
| 04 | [VAE](04-vae.md) | A Variational Autoencoder — architektúra, tenzorméretek, miért kell |
| 05 | [Navigation](05-navigation.md) | Útvonaltervezés: A\*, gráfépítés, RoadOption, kanyar-döntés |
| 06 | [Fogalomtár](06-fogalomtar.md) | Minden szakkifejezés magyarul: RL, CARLA, VAE, tervezés |
| 07 | [Továbbfejlesztés](07-tovabbfejlesztes.md) | Ismert hibák, receptek a módosításhoz, ROS2 terv, hibakeresés |

## Hogyan olvasd?

**Ha most ismerkedsz a projekttel:** 00 → 01 → 02, a 06-ot pedig tartsd nyitva mellette
szótárként.

**Ha módosítani akarsz:** 07, ott vannak a konkrét receptek (új jutalom, új állapotelem,
fék hozzáadása, térképváltás, eval script).

**Ha valami nem működik:** 07, 5. fejezet (hibakeresési checklist) és 1. fejezet
(ismert hibák).

## Jelölések

- ★ = különösen fontos rész
- ⚠ = figyelmeztetés, buktató
- 🔴🟡🟢 = hibák súlyossága a 07-es fájlban
- 📍 = pontos helymegjelölés a kódban (kattintható)
