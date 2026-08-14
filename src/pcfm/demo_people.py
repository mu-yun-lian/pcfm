from __future__ import annotations


DEMO_SEED_VERSION = "verified-demos-v2"


DEMO_PEOPLE: tuple[dict[str, object], ...] = (
    {
        "person_id": "demo-sally-ride",
        "name": "Sally Ride",
        "aliases": ["Sally K. Ride", "Dr. Sally Ride"],
        "language": "en",
        "identity_note": "American physicist and NASA astronaut (1951-2012)",
        "focus_domain": "NASA astronaut selection, Shuttle operations, and space-program planning",
        "time_start": "1977-01-01",
        "time_end": "2002-12-06",
        "avatar": "/demo-sally-ride.svg",
        "description": "演示人物；基于 NASA Johnson Space Center 口述史逐字稿。探索性预测，准确性尚未验证。",
        "recommended_questions": [
            {
                "kind": "direct",
                "label": "资料内直接问题",
                "text": "Tell us what prompted you to write that note and describe the events that followed.",
            },
            {
                "kind": "nearby",
                "label": "相近问题（需更多取向资料）",
                "text": "How did NASA's selection process and working culture shape your early experience as an astronaut?",
            },
            {
                "kind": "out_of_scope",
                "label": "应当拒答",
                "text": "What investment strategy would you use for cryptocurrency in 2026?",
            },
        ],
        "sources": [
            {
                "title": "NASA JSC Oral History - Sally K. Ride, 22 October 2002",
                "speaker": "Sally K. Ride",
                "source_date": "2002-10-22",
                "dataset_role": "model_source",
                "content_authenticity": "verbatim_transcript",
                "source_url": "https://www.nasa.gov/wp-content/uploads/2025/08/ridesk-10-22-02.pdf",
                "source_locator": "PDF pp. 1-6; speaker turns labelled RIDE",
                "source_context": "NASA Johnson Space Center Oral History Project interview by Rebecca Wright in San Diego; transcript notes that Ride amended answers for clarification. Stored turns are verbatim excerpts and do not claim to reproduce the complete interview.",
                "original_language": "en",
                "text": """Q: Tell us what prompted you to write that note and describe the events that followed.
A: I saw an ad in the Stanford University student newspaper that the Center for Research on Women at Stanford had put in the paper on behalf of NASA. It announced that NASA was accepting applications for what would be the astronaut class of 1978. The ad made it clear that NASA was looking for scientists and engineers, and it also made it clear that they were going to accept women into the astronaut corps.

Q: And what was the next step? How long was it before you heard back from the selection office that they wanted you to apply?
A: Well, it wasn't very long. The selection office had a pretty good process in place even back then. It struck me as kind of an entertaining process. I remember that relatively quickly—and I don't know whether that was a week or a month—I got a simple one- or two-page application.

Q: How did you learn that you were selected to be a candidate?
A: I got a phone call from George W. S. Abbey very, very early in the morning California time. He was probably going through his list, making calls to those selected at, oh, seven-thirty or eight in the morning, Houston time. I was awakened by the phone call, and when I heard George Abbey on the phone, I thought it was probably good news.

Q: And your reaction?
A: Well, it took me a while to wake up! I thought maybe I was dreaming. But, of course, I was thrilled. My biggest frustration was that it was five or six in the morning in California, so all my friends and family were asleep.

Q: Did this impact on your private life affect your decision or make you think that you were getting into something a little bit more?
A: Actually, it didn't. It wasn't particularly burdensome after the initial flurry of interviews. There was a fair amount of it, but it was still easy to have a normal life.

Q: It was a class of thirty-five, and a lot of attention was on the fact that it did include six females. How was the rest of the class impacted by the fact that so much of the attention was on six members instead of the whole thirty-five?
A: I think the rest of the class understood that that was natural and maybe even appreciated it! It was really a good group of thirty-five. The selection committee was looking for men that were comfortable working with women, that were used to working with women, and that had no problem working with women, and they succeeded.""",
            },
            {
                "title": "NASA JSC Oral History 2 - Sally K. Ride, 6 December 2002",
                "speaker": "Sally K. Ride",
                "source_date": "2002-12-06",
                "dataset_role": "final_holdout",
                "content_authenticity": "verbatim_transcript",
                "source_url": "https://www.nasa.gov/wp-content/uploads/2025/08/ridesk-12-6-02.pdf",
                "source_locator": "PDF pp. 3-4; speaker turns labelled RIDE",
                "source_context": "Part two of the NASA Johnson Space Center Oral History Project interview by Rebecca Wright in San Antonio; entire later interview is excluded from training and style distillation. Stored turns are verbatim excerpts.",
                "original_language": "en",
                "text": """Q: How was your report received by colleagues?
A: It was received very well. We testified before Congress, and we briefed it widely to the National Research Council, the President's Science Advisor, and a variety of other groups. There were several things that came out of it. One was NASA's Mission to Planet Earth; another was the Office of Exploration.

Q: Could you possibly tell us what your most challenging milestone was while you were working with the space program?
A: I think my biggest challenge was just trying to breathe right after the engines ignited on my first launch! It's hard to say what my most challenging milestone was. The space program is wonderful in that it is a series of challenges and a series of very interesting and very rewarding experiences.""",
            },
        ],
    },
    {
        "person_id": "demo-barack-obama",
        "name": "Barack Obama",
        "aliases": ["President Barack Obama", "President Obama"],
        "language": "en",
        "identity_note": "44th President of the United States; official White House transcript context",
        "focus_domain": "United States foreign policy and public presidential press conferences",
        "time_start": "2015-07-15",
        "time_end": "2016-12-16",
        "avatar": "/demo-barack-obama.svg",
        "description": "演示人物；基于美国白宫官方归档记者会逐字稿。探索性预测，准确性尚未验证。",
        "recommended_questions": [
            {
                "kind": "direct",
                "label": "资料内直接问题",
                "text": "What steps will you take to enable a more moderate Iran? And does this deal allow you to more forcefully counter Iran's destabilizing actions in the region quite aside from the nuclear question?",
            },
            {
                "kind": "nearby",
                "label": "相近问题（需更多取向资料）",
                "text": "How should a president weigh an imperfect diplomatic agreement against the risks of military action?",
            },
            {
                "kind": "out_of_scope",
                "label": "应当拒答",
                "text": "Which consumer smartphone should I buy in 2026?",
            },
        ],
        "sources": [
            {
                "title": "White House Press Conference by the President, 15 July 2015",
                "speaker": "Barack Obama",
                "source_date": "2015-07-15",
                "dataset_role": "model_source",
                "content_authenticity": "verbatim_transcript",
                "source_url": "https://obamawhitehouse.archives.gov/the-press-office/2015/07/15/press-conference-president/",
                "source_locator": "East Room transcript, 1:25 P.M. EDT; Q&A turns for Andrew Beatty, Jon Karl, Carol Lee, Michael Crowley, and Major Garrett",
                "source_context": "Official White House Office of the Press Secretary transcript. Stored turns are verbatim, speaker-attributed excerpts and do not claim to reproduce each complete answer; the complete source remains linked.",
                "original_language": "en",
                "text": """Q: What steps will you take to enable a more moderate Iran? And does this deal allow you to more forcefully counter Iran's destabilizing actions in the region quite aside from the nuclear question?
A: The starting premise of our strategy with respect to Iran has been that it would be a grave threat to the United States and to our allies if they obtained a nuclear weapon. And so everything that we've done over the last six and a half years has been designed to make sure that we address that number-one priority.

Q: Does it give you any pause to see this deal praised by Syrian dictator Assad as a great victory for Iran, or praised by those in Tehran who still shout death to America, and yet our closest ally in the Middle East calls it a mistake of historic proportions?
A: It does not give me pause that Mr. Assad or others in Tehran may be trying to spin the deal in a way that they think is favorable to what their constituencies want to hear. That's what politicians do. Well, now we have a document so you can see what the deal is.

Q: Prime Minister Netanyahu said that you have a situation where Iran can delay 24 days before giving access to military facilities.
A: I'm happy to—that's a good example. So let's take the issue of 24 days. This has been I think swirling today, the notion that this is insufficient in terms of inspections. Now, keep in mind, first of all, that we'll have 24/7 inspections of declared nuclear facilities.

Q: I want to ask you about the arms and ballistic missile embargo. Why did you agree to lift those even with the five- and eight-year durations?
A: So the issue of the arms embargo and ballistic missiles is of real concern to us—has been of real concern to us. And it is in the national security interest of the United States to prevent Iran from sending weapons to Hezbollah, for example, or sending weapons to the Houthis in Yemen that accelerate a civil war there.

Q: Many analysts believe that a negotiated political settlement in Syria will require working directly with Iran and giving Iran an important role. Do you agree? And what about the fight against ISIS? What would it take for explicit cooperation between the U.S. and Iran?
A: I do agree that we're not going to solve the problems in Syria unless there's buy-in from the Russians, the Iranians, the Turks, our Gulf partners. It's too chaotic. There are too many factions. There's too much money and too many arms flooding into the zone.

Q: Can you tell the country why you are content, with all the fanfare around this deal, to leave the conscience of this nation and the strength of this nation unaccounted for in relation to these four Americans?
A: I got to give you credit, Major, for how you craft those questions. The notion that I am content as I celebrate with American citizens languishing in Iranian jails—Major, that's nonsense, and you should know better. I've met with the families of some of those folks. Nobody is content.""",
            },
            {
                "title": "White House Joint Press Conference, 2 August 2016 - Trump fitness assessment",
                "speaker": "Barack Obama",
                "source_date": "2016-08-02",
                "dataset_role": "model_source",
                "content_authenticity": "verified_quote",
                "source_url": "https://obamawhitehouse.archives.gov/the-press-office/2016/08/02/remarks-president-obama-and-prime-minister-lee-singapore-joint-press",
                "source_locator": "East Room transcript; first reporter question; answer opening sentence",
                "source_context": "Official White House Office of the Press Secretary transcript. The stored question clause and answer sentence are exact excerpts used as one bounded public response episode.",
                "original_language": "en",
                "entity_aliases": [
                    "Donald Trump",
                    "Donald J. Trump",
                    "Trump",
                    "唐纳德·特朗普",
                    "特朗普",
                ],
                "text": """Q: Does it make you question his fitness to be President?
A: Yes, I think the Republican nominee is unfit to serve as President.""",
            },
            {
                "title": "White House Press Conference by the President, 16 December 2016",
                "speaker": "Barack Obama",
                "source_date": "2016-12-16",
                "dataset_role": "final_holdout",
                "content_authenticity": "verbatim_transcript",
                "source_url": "https://obamawhitehouse.archives.gov/the-press-office/2016/12/16/press-conference-president/",
                "source_locator": "James S. Brady Press Briefing Room transcript, 2:40 P.M. EST; opening Russia and election Q&A turns",
                "source_context": "Official White House Office of the Press Secretary transcript; entire later press conference is excluded from training and style distillation. Stored turns are verbatim excerpts.",
                "original_language": "en",
                "text": """Q: Can you, given all the intelligence that we have now heard, assure the public that this was, once and for all, a free and fair election? And specifically on Russia, do you feel any obligation now to show the proof and declassify some of the intelligence?
A: I can assure the public that there was not the kind of tampering with the voting process that was of concern and will continue to be of concern going forward; that the votes that were cast were counted, they were counted appropriately. We will provide evidence that we can safely provide that does not compromise sources and methods.

Q: Are you prepared to call out President Putin by name for ordering this hacking? And is the administration's quarreling with the incoming team tarnishing the transition of power?
A: Well, first of all, with respect to the transition, I think they would be the first to acknowledge that we have done everything we can to make sure that they are successful as I promised. And that will continue. What we've simply said is the facts, which are that, based on uniform intelligence assessments, the Russians were responsible for hacking the DNC.""",
            },
        ],
    },
)
