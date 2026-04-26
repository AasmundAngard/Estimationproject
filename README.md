# Estimationproject
Repo for kode til TTT4275 Estmering, Klassifiseringsprosjekt 6 - musikk

## Hvordan jobbe med github

Vi bruker forskjellige brancher til å legge til og flytte på endringer.

### Nyttige git-kommandoer
git branch - viser oversikt over alle brancher på din pc, og markerer den du er i nå

git checkout navnpåbranch - flytter deg over i navnpåbranch. Du må huske å commite endringene dine før du bytter branch.

git pull - henter endringer fra github for branchen du er i nå

git merge navnpåbranch - henter endringer fra din lokale navnpåbranch til branchen du er i nå. Kan trigge en konflikt du må løse opp i.

### Hvordan legge til egne endringer
1. Gjør endringer/skriv noe i din egen branch.
2. Når du er fornøyd, skriv en commit-kommentar og trykk commit i git-extensionen (den med grenene).
3. git checkout main (i terminalen)
4. git pull (i terminalen)
5. git merge dinbranch
6. Dersom du fikk konflikt: manuelt rydd opp i konflikten i filene som markeres i git-extensionen, lagre, og trykk deretter på (+) på filene og ''continue''. Du får opp konfliktene i git-extensionen, ved å trykke på filene som er listet opp der.
7. git push
8. Dersom du mottok nye endringer idet du kjørte git pull: git checkout dinbranch og git merge main

Oppsummering: hent andres endringer fra github til din lokale main, hent så endringene fra din arbeidsbranch til main, løs sammenfletting av endringer manuelt, og last opp resultatet tilbake til github.

Dersom du mottok nye endringer fra github, husk å hente endringene tilbake til din branch.



### Hvordan hente andres endringer
1. git checkout main
2. git pull
3. git checkout dinbranch
4. git merge main
5. Hvis konflikter, må du manuelt fikse konfliktene ved å gjennom filene som markeres i git-extensionen, og velge hvilken av de markerte kodebitene du skal ha med.
6. Dersom du måtte resolve: Når ferdig med resolving, trykk på (+) for filene, og ''continue'' i git-extensionen. 

Oppsummert: Endringer fra andre skal komme gjennom main. Bruk git pull i main for å hente fra github. Bruk git merge main når i din egen branch for å legge endringer i din branch, og løs manuelt konflikter.
