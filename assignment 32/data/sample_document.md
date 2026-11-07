# Halcyon Ridge Observatory: Operations and Research Handbook

Edition 4.2. This handbook is a fictional document written for a retrieval-augmented generation exercise. Every name, number and procedure in it is invented, so a language model cannot answer questions about it from memory and must rely on the retrieved text.

## 1. Introduction and History

Halcyon Ridge Observatory sits at an altitude of 2,870 metres on the eastern shoulder of the Meridian Range, roughly 140 kilometres from the nearest coastal city. The site was first surveyed in 1987 by a team from the Coastal Institute of Astronomy, who measured median seeing of 0.71 arcseconds over three winters of testing. That figure, unusually good for a continental site, persuaded a consortium of four universities to fund construction.

Construction of the access road began in 1991 and the first telescope, a 1.2-metre reflector now called the Pathfinder, saw first light on 14 March 1994. The flagship instrument, the 3.6-metre Aurelia Telescope, followed in 2003 after a nine-year build that was delayed twice: once by a landslide that closed the access road for an entire summer, and once by a flaw discovered in the first primary mirror blank, which had to be recast. The recast blank was polished over twenty-two months by a specialist optical workshop and delivered by a convoy of trucks that drove at walking pace for the final 30 kilometres.

Today the observatory is operated by the Halcyon Ridge Consortium, which has grown to seven member institutions. About 60 people work on the mountain during a typical week, including telescope operators, engineers, cooks, and a small medical team. A further 25 staff work at the sea-level support office in Port Calloway, where data reduction, administration, and most software development take place.

The observatory's mission statement, adopted in 2010, commits the site to three goals: to provide world-class observing conditions to consortium astronomers, to protect the darkness of the night sky above the Meridian Range, and to train the next generation of observational scientists. The third goal is the reason the observatory runs a residency programme for graduate students, described in Section 9.

## 2. Site and Facilities

The summit area covers 42 hectares. Its main buildings are the Aurelia enclosure, the Pathfinder dome, the Control Building, the Residence, and the Workshop. A smaller structure called the Seeing Tower houses the site monitoring instruments described in Section 5.

The Control Building is the operational heart of the mountain. It contains the main control room, a remote-observing suite with four workstations, the data centre, and the shift supervisor's office. The control room is deliberately kept dim during the night, with red lighting only, so that operators retain their night vision when they step outside to inspect the sky.

The Residence provides 48 single bedrooms, a dining hall, a small library, and a lounge. Meals are served at fixed times: breakfast is served from 07:00 to 09:00, lunch from 12:00 to 13:30, and dinner from 18:00 to 19:30. A "night lunch" is left in the control room kitchen at 23:00 for observers and operators who are working through the night. Because many residents sleep during the day, the corridors of the Residence are designated quiet zones between 08:00 and 16:00, and blackout curtains are fitted in every bedroom.

The Workshop houses mechanical and electronic repair benches, a clean room for handling optics, and the aluminising chamber. The aluminising chamber is used every two years to strip and recoat the Aurelia primary mirror with a fresh reflective layer. During a recoating campaign, the Aurelia Telescope is out of service for about three weeks, and the campaign is always scheduled in February, which is historically the cloudiest month on the ridge.

Power is supplied by a dedicated line from the regional grid, backed up by two diesel generators that can each carry the full summit load. A 400-kilowatt-hour battery bank bridges the gap of roughly 40 seconds between a grid failure and generator start-up, so that the telescope drives and the data centre never lose power. Water is trucked up the mountain twice a week and stored in two 60,000-litre tanks.

## 3. Telescopes and Instruments

### 3.1 The Aurelia Telescope

The Aurelia Telescope is a 3.6-metre Ritchey-Chrétien reflector on an altitude-azimuth mount. Its primary mirror is made of a low-expansion glass ceramic and weighs 11.2 tonnes. The mirror is supported by 84 actuators that adjust its shape every few seconds under the control of the active optics system, compensating for the way the glass sags as the telescope tilts. The secondary mirror can be tilted quickly to correct for wind shake.

The telescope has two Nasmyth foci and one Cassegrain focus. Switching between instruments at the two Nasmyth foci takes under five minutes, because the tertiary mirror simply rotates to direct light to one side or the other. Changing the Cassegrain instrument, in contrast, requires a crane and a full day of engineering work, so the Cassegrain instrument is normally changed only during scheduled maintenance blocks.

### 3.2 Instrument Suite

The following instruments are currently offered to observers on the Aurelia Telescope:

- KESTREL, a wide-field optical imager at Nasmyth focus A. It has a 32-arcminute field of view and an eight-slot filter wheel holding the standard broadband filters and three narrowband filters.
- HERON, a high-resolution echelle spectrograph at Nasmyth focus B, with a resolving power of 85,000. HERON is housed in a temperature-stabilised room held to within 0.01 degrees Celsius, which allows it to measure stellar radial velocities with a precision of about 1.5 metres per second.
- OSPREY, a near-infrared imager and spectrograph at the Cassegrain focus, cooled with liquid nitrogen to minus 196 degrees Celsius. OSPREY's detector must be warmed up slowly, at no more than 2 degrees per minute, whenever the instrument is removed.
- WREN, a visitor port at Nasmyth focus B that allows external teams to mount their own instruments for up to 30 nights per semester, subject to a safety review.

The Pathfinder telescope carries a single instrument, a fast photometer called SWIFT, which is used mainly for monitoring variable stars and for training students. Pathfinder is fully robotic and can run unattended for an entire night once a schedule has been uploaded.

### 3.3 Instrument Calibration

Every instrument has a calibration plan that must be completed at the start and end of each night. For KESTREL, this means twilight flat fields in every filter used that night and a set of eleven bias frames. For HERON, observers take a thorium-argon arc lamp exposure before and after each science target, because the spectrograph's wavelength solution drifts slightly as the building cools during the night. Calibration frames are archived alongside the science data and are publicly released at the same time.

## 4. Observing Time Allocation

Observing time on the Aurelia Telescope is divided into two semesters per year. Semester A runs from 1 February to 31 July and Semester B runs from 1 August to 31 January. Proposals for each semester are submitted through the Consortium's online portal.

The proposal deadline for Semester A is 15 September of the preceding year, and the deadline for Semester B is 15 March. Late proposals are not accepted under any circumstances, although the Director may grant a small amount of discretionary time, described below.

All proposals are reviewed by the Time Allocation Committee, known on the mountain as the TAC. The TAC has nine members: one appointed by each of the seven member institutions, plus two external astronomers from outside the consortium who serve rotating three-year terms. Each proposal is graded on scientific merit, technical feasibility, and the likelihood that the results will be published. A typical semester is oversubscribed by a factor of about 3.4, meaning that astronomers request roughly three and a half times more nights than are available.

Time is shared among the consortium partners in proportion to their financial contribution, but the TAC ranks proposals purely on merit within each partner's share. Five percent of all Aurelia time is reserved as Director's Discretionary Time. Requests for Director's Discretionary Time can be submitted at any point in the semester and are intended for urgent targets such as newly discovered supernovae, comets, or the optical counterparts of gravitational-wave events. The Director aims to answer such requests within 48 hours.

Programmes that are awarded time are classified as either classical or queue. In classical mode, the astronomer travels to the mountain, or connects to the remote-observing suite, and runs their own observations on assigned nights. In queue mode, the astronomer submits detailed observing blocks and the night staff execute them whenever conditions match the requested constraints. About two thirds of the time on HERON is now scheduled in queue mode, because high-precision radial velocity programmes benefit from observations spread evenly through the semester.

## 5. Weather, Site Monitoring, and Operating Limits

### 5.1 Monitoring Instruments

The Seeing Tower carries a differential image motion monitor, which measures atmospheric seeing every two minutes, a cloud camera that photographs the whole sky every 60 seconds, a precipitation sensor, and a weather mast. The weather mast stands 20 metres tall and records wind speed, wind direction, temperature, humidity, and dew point. All of these data are displayed on a large screen in the control room and archived for later analysis.

The site's long-term statistics show that about 68 percent of nights are usable for observing, and about 52 percent are fully photometric, meaning that the sky is completely clear for the whole night. The median seeing measured since 2005 is 0.78 arcseconds, slightly worse than the original survey value, which most staff attribute to the growth of a nearby town.

### 5.2 Dome Operating Limits

The safety of the telescope and its optics takes priority over any observing programme. The operator on duty has the authority to close the dome whenever conditions approach the limits below, and no observer may overrule that decision.

The Aurelia dome must be closed when the sustained wind speed exceeds 18 metres per second, or when gusts exceed 22 metres per second. If the sustained wind speed lies between 14 and 18 metres per second, the dome may stay open, but the telescope must not point within 30 degrees of the direction the wind is blowing from, to reduce wind shake on the mirror.

The dome must also be closed when relative humidity exceeds 85 percent, when the difference between the air temperature and the dew point falls below 2 degrees Celsius, or when any precipitation is detected. After a closure caused by humidity, the dome may only reopen once humidity has stayed below 80 percent for at least 30 continuous minutes. This hysteresis prevents the dome from opening and closing repeatedly on marginal nights, which wears out the shutter motors.

The smaller Pathfinder dome has slightly more relaxed limits: it closes automatically when the sustained wind exceeds 20 metres per second or humidity exceeds 90 percent. Because Pathfinder is robotic, these closures are performed by software without operator intervention, although the operator is notified.

### 5.3 Temperature Management

Mirror seeing, which is the blurring caused by a mirror that is warmer than the surrounding air, is one of the largest sources of image degradation at any observatory. To control it, the Aurelia enclosure is air-conditioned during the day to the temperature predicted for the start of the coming night. The prediction is produced each afternoon at 15:00 by a forecasting model that uses the previous three nights of weather data. The goal is to keep the primary mirror within 0.5 degrees Celsius of the ambient air temperature throughout the night.

The data centre, by contrast, is cooled to a constant 21 degrees Celsius all year round. The kitchen freezer must be kept at minus 18 degrees Celsius, and the medical refrigerator in the clinic at between 2 and 8 degrees Celsius. These figures are mentioned here because staff sometimes confuse the various temperature alarms that appear on the facilities dashboard.

## 6. Night Operations

### 6.1 Roles

Every night at least two people must be in the Control Building: the telescope operator, who is responsible for the safety of the telescope and the dome, and either the visiting astronomer or a queue observer, who is responsible for the science. A shift supervisor is on call from the Residence and must be able to reach the Control Building within ten minutes. Nobody may work alone at the summit at night; this is known as the two-person rule and it applies to all outdoor and technical work after sunset.

### 6.2 Start-of-Night Procedure

The start-of-night procedure begins two hours before sunset. The operator follows these steps in order:

1. Review the weather forecast and the cloud camera images, and record the expected conditions in the night log.
2. Confirm that the instrument for the first programme is mounted, cooled, and has passed its health checks.
3. Open the dome louvres one hour before sunset to begin flushing warm air out of the enclosure.
4. Open the main dome shutter at sunset, provided all operating limits are satisfied, and begin twilight calibrations.
5. Point the telescope at a bright standard star, run the active optics initialisation sequence, and confirm that image quality is within 0.2 arcseconds of the seeing reported by the monitor.
6. Hand over to the observer and record the time of handover in the night log.

If image quality cannot be brought within the required tolerance after three attempts at step 5, the operator must contact the on-call optics engineer before science observations begin.

### 6.3 End-of-Night Procedure

Observing ends when the Sun reaches 12 degrees below the horizon in the morning, which is known as nautical twilight. The operator then parks the telescope at the zenith, closes the dome, and runs the end-of-night calibrations. The night log must be completed and submitted before the operator goes off shift; it records weather, the time lost to technical faults, the time lost to weather, and a summary of the programmes observed.

## 7. Data Management

Raw data from every instrument are written to the summit data centre and copied within one hour to the archive at the Port Calloway support office over a dedicated microwave link. A second copy is sent each week on encrypted disks by road, so that the archive survives even if the microwave link is down for an extended period.

Data belong to the principal investigator of the programme for a proprietary period of 18 months from the date of observation. After the proprietary period ends, the data, together with their calibration frames, are released to the public through the Halcyon Ridge Science Archive. Principal investigators may request a one-time extension of six months if a graduate student's thesis depends on the data, but such extensions must be approved by the Director.

The archive currently stores about 1.9 petabytes of data. A pipeline called LUMEN processes all KESTREL and HERON data automatically each morning, producing calibrated images and extracted spectra that observers can download by 10:00 local time. OSPREY data are not yet processed by LUMEN; observers must reduce them with their own software.

## 8. Health and Safety

### 8.1 Altitude

At 2,870 metres, the air holds about 30 percent less oxygen than at sea level. All new arrivals must spend their first night at the Midway Lodge at 1,600 metres before travelling to the summit, which allows the body to begin acclimatising. Visitors with heart or lung conditions must obtain medical clearance before their first visit. Mild symptoms of altitude sickness, such as headache and poor sleep, are common during the first two days and are usually treated with rest and fluids.

Any person showing severe symptoms of altitude sickness, such as confusion, loss of coordination, or breathlessness while resting, must be given supplemental oxygen immediately and driven down to the Midway Lodge without waiting for the symptoms to improve. Descent is the single most effective treatment, and the medical team is instructed never to delay it in order to observe the patient.

### 8.2 Driving

The access road is 38 kilometres long, unpaved for the final 11 kilometres, and includes 46 hairpin bends. Only staff who have completed the observatory's mountain-driving course may drive on it. Between sunset and sunrise, all vehicles approaching the summit must drive with sidelights only for the last 2 kilometres, to avoid shining headlights into the domes. Snow chains must be carried in every vehicle from November to April.

### 8.3 Working at Height and Cryogens

Any work more than 2 metres above the floor inside the Aurelia enclosure requires a harness and a second person acting as a spotter. Liquid nitrogen for OSPREY is handled only by trained staff wearing face shields and cryogenic gloves, and the nitrogen fill is never performed by a single person, in keeping with the two-person rule.

### 8.4 Fire

The summit is far from the nearest fire station, so fire prevention is taken especially seriously. Smoking is forbidden anywhere on the site. Fire drills are held at the start of each semester. In the event of a fire alarm at night, the telescope operator closes the dome and parks the telescope only if this can be done in less than two minutes; otherwise the operator leaves immediately and the telescope is abandoned. The assembly point for the Control Building is the helipad, while the assembly point for the Residence is the car park.

## 9. Graduate Residency Programme

The Graduate Residency Programme invites up to six doctoral students each year to live at the observatory for a period of three to six months. Residents receive a stipend, free accommodation, and meals, and in exchange they work as queue observers for roughly half of their nights. The remaining nights are available for their own thesis projects, which can use the Pathfinder telescope without going through the TAC.

Applications for the residency open on 1 May and close on 30 June each year. A selection panel made up of three staff astronomers and one former resident reviews the applications. Successful applicants are notified by the end of August and normally begin their residency in the following January.

Every resident is assigned a mentor from the staff astronomers. At the end of the residency, each student gives a public talk in Port Calloway about their research, a tradition that has become one of the most popular outreach events in the region.

## 10. Dark Sky Protection and Outreach

Protecting the night sky is part of the observatory's mission. Since 2012, Halcyon Ridge has been the core of the Meridian Dark Sky Reserve, which covers 3,200 square kilometres. Within the reserve, outdoor lighting must be fully shielded so that no light is emitted above the horizontal, and must have a colour temperature of 2,700 kelvin or lower. The observatory employs a dark-sky officer who works with surrounding municipalities to monitor compliance, and who measures sky brightness each month at twelve fixed locations around the reserve.

Public visits to the summit are offered on Saturday afternoons between April and October. Visitors tour the Aurelia enclosure and the control room but must leave the summit by sunset. Night-time public observing takes place instead at the Visitor Centre at Midway Lodge, which has its own 0.6-metre telescope and a planetarium. School groups can book free visits to the Visitor Centre on weekdays.

## 11. Maintenance

The Aurelia Telescope has two kinds of scheduled maintenance. Short maintenance blocks of one night each occur every month around full moon, when the bright sky makes some observations impossible. Longer maintenance shutdowns occur once per year, during which the Cassegrain instrument may be exchanged and major mechanical work is performed.

The primary mirror is cleaned with carbon dioxide snow once a month, which removes dust without touching the surface. As already noted, the mirror is fully recoated every two years in February. Reflectivity of the mirror is measured weekly; if it drops below 84 percent at visible wavelengths, the engineering team may bring a recoating forward.

Unscheduled faults are logged in the observatory's fault-tracking system. Faults are classified by severity: a Severity 1 fault stops all observing on the telescope; a Severity 2 fault prevents the use of one instrument; and a Severity 3 fault is a minor problem that has a workaround. For a Severity 1 fault occurring at night, the on-call engineer must be called immediately and must be on site within 45 minutes. Severity 2 faults are handled the next working morning, and Severity 3 faults are scheduled into the next maintenance block.

## 12. Governance

The observatory is governed by the Consortium Board, which meets twice a year and includes one representative from each member institution. The Board appoints the Director for a five-year term, renewable once. The Director is responsible for the day-to-day operation of the observatory and reports to the Board.

New institutions may join the consortium by contributing at least 4 percent of the annual operating budget for a minimum of five years. Any decision to admit a new member requires a two-thirds majority of the Board. Withdrawal from the consortium requires two years' notice.

The annual operating budget is currently about 14 million in the consortium's accounting currency. Roughly 45 percent is spent on staff, 20 percent on instruments and upgrades, 15 percent on power, water, and transport, 10 percent on the support office, and the remaining 10 percent on outreach, training, and contingency.

## Appendix A: Glossary

Seeing: the blurring of star images caused by turbulence in the atmosphere, usually expressed as the angular width of a star image in arcseconds.

Photometric night: a night on which the sky is completely free of clouds from dusk to dawn, so that brightness measurements can be calibrated absolutely.

Queue observing: a mode in which night staff carry out observations on behalf of astronomers, choosing which programme to observe based on current conditions and programme priorities.

Hysteresis: the use of different thresholds for closing and reopening, so that a system does not switch back and forth rapidly when a measurement hovers near a single limit.

Nasmyth focus: a focus position on the side of an altitude-azimuth telescope, reached by a flat tertiary mirror, where heavy instruments can be mounted on a stable platform.
