# Example summaries (fine-tuned 16-bit versus 2-shot Gemini)

Texts are cut to 700 characters. Chosen by the largest per-bill ROUGE-L difference, not hand-picked.

## Largest gains for the fine-tuned model

### Test bill 16 (7,976 characters), ROUGE-L fine-tuned 0.667, 2-shot Gemini 0.206

**Reference:** Amends the Internal Revenue Code concerning:  (1) the treatment of bonds issued to finance electric output facilities; and (2) the special rules for nuclear decommissioning costs.

**Fine-tuned:** Amends the Internal Revenue Code to: (1) repeal the treatment of bonds issued to finance electric output facilities as private activity bonds; and (2) limit the amount of nuclear decommissioning costs that may be paid into the Nuclear Waste Fund.

**2-shot Gemini:** Amends the Internal Revenue Code regarding the tax treatment of bonds issued to finance electric output facilities and nuclear decommissioning costs.   For "pre-effective date" electric output facility bonds—defined as those issued before the enactment of a comprehensive electricity competition act—it specifies that their status as private activity bonds will be determined without regard to "permissible competitive actions" taken by the issuer, such as implementing retail competition or providing nondiscriminatory access to transmission facilities. For bonds issued on or after the enactment of such a competition act, it classifies bonds used for electric output facilities (excluding small di

### Test bill 147 (6,703 characters), ROUGE-L fine-tuned 0.706, 2-shot Gemini 0.316

**Reference:** Fixed Asset Relief Act of 2015 This bill amends the Internal Revenue Code to increase the additional depreciation allowance (bonus depreciation) from 50% to 100% of the adjusted basis of qualifying business property and to make such increased allowance permanent. 

**Fine-tuned:** Fixed Asset Relief Act of 2015 - Amends the Internal Revenue Code to increase the bonus depreciation percentage for qualified property from 50% to 100%. Makes such increase permanent.

**2-shot Gemini:** Fixed Asset Relief Act of 2015 - Amends the Internal Revenue Code to increase the bonus depreciation allowance for qualified property from 50 percent to 100 percent and make this deduction permanent for property placed in service after December 31, 2014. Defines "qualified property" as property with a recovery period of 20 years or less, computer software, water utility property, or qualified leasehold improvement property, provided the original use commences with the taxpayer. Excludes property subject to the alternative depreciation system. Provides for special rules regarding sale-leaseback transactions and syndications to determine when property is considered originally placed in service

### Test bill 41 (5,181 characters), ROUGE-L fine-tuned 0.763, 2-shot Gemini 0.374

**Reference:** RCAF/RAF-Americans Congressional Gold Medal Act This bill directs the Speaker of the House of Representatives and the President pro tempore of the Senate to arrange for the award of a Congressional Gold Medal in commemoration of all U.S. nationals who joined the Royal Canadian Air Force or the Royal Air Force during World War II.

**Fine-tuned:** RCAF/RAF-Americans Congressional Gold Medal Act - Authorizes the Speaker of the House of Representatives and the President pro tempore of the Senate to make appropriate arrangements for the presentation, on behalf of the Congress, of a single gold medal in commemoration of all U.S. nationals who joined the Royal Canadian Air Force or the Royal Air Force during World War II, both before and after Japan's attack on Pearl Harbor, in recognition of their contributions to the Nation.

**2-shot Gemini:** RCAF/RAF-Americans Congressional Gold Medal Act - Expresses Congressional gratitude to the U.S. nationals who volunteered for the Royal Canadian Air Force (RCAF) and the Royal Air Force (RAF) during World War II. Authorizes the Speaker of the House of Representatives and the President pro tempore of the Senate to arrange for the presentation of a single Congressional Gold Medal in their honor. Requires the Secretary of the Treasury to strike the gold medal, which will subsequently be housed at the Smithsonian Institution for display and research. Encourages the Smithsonian to display the medal at other appropriate locations associated with these veterans. Authorizes the Secretary to strike a

## Largest losses for the fine-tuned model

### Test bill 89 (7,925 characters), ROUGE-L fine-tuned 0.120, 2-shot Gemini 0.429

**Reference:** Temporarily designates Poland as a program country for purposes of the Immigration and Nationality Act's visa waiver program (VWP), notwithstanding designation requirements of current law.  Requires the Secretary of Homeland Security to determine the nonimmigrant visa overstay rate for Polish nationals for the elapsed period of temporary designation and, based on such calculation, to determine whether Poland shall be designated permanently but conditionally for VWP purposes.  States that, if permanently but conditionally designated, Poland shall remain a VWP country as long as the annual overstay rate remains below three percent. Requires the Secretary to: (1) suspend VWP designation for one

**Fine-tuned:** Polish-American Community of 9.3 Million People Contributed Significantly to the Development of the United States. Since the movement known as ``Solidarity'' and the turning point of 1989, Poland has undergone great political, social, and economic changes. It has become a nation committed to traditional values, rules of law, freedom, and democracy.

**2-shot Gemini:** Designates Poland as a participant in the Visa Waiver Program (VWP) for a temporary period of 18 months, notwithstanding existing statutory requirements under the Immigration and Nationality Act. Directs the Secretary of Homeland Security to review Poland’s overstay rate within one year of enactment to determine whether the country should be granted permanent, conditional designation. Under this conditional status, Poland’s participation in the VWP remains contingent upon maintaining an annual overstay rate of less than 3 percent. Provides for the suspension of VWP privileges if the overstay rate exceeds 3 percent in a single fiscal year, and mandates termination of the designation if the ra

### Test bill 142 (6,408 characters), ROUGE-L fine-tuned 0.222, 2-shot Gemini 0.478

**Reference:** North Korean Counterterrorism and Nonproliferation Act - Provides for the continuation of restrictions against the government of North Korea unless the President certifies to Congress that North Korea has met certain benchmarks respecting: (1) missile or nuclear technology transfers; (2) support of terrorist groups and terrorist activities, (3) counterfeiting of U.S. currency, (4) release of South Korean POWs, Japanese journalists, and Kim Donk-Shik; and (5) Bureau 39's closure.

**Fine-tuned:** North Korean Counterterrorism and Nonproliferation Act - Directs the President to certify to Congress that the Government of North Korea: (1) is no longer engaged in the illegal transfer of missile or nuclear technology, particularly to the Governments of Iran, Syria, or any other country, the government of which the Secretary of State has determined, for purposes of any of the provisions of law specified in the matter preceding this paragraph, is a government that has repeatedly provided support for acts of international terrorism; (2) is no longer engaged in training, harboring, supplying, financing, or supporting in any way: (A) Hamas, Hezbollah, or the Japanese Red Army, or any member of

**2-shot Gemini:** North Korean Counterterrorism and Nonproliferation Act - Establishes requirements for the continuation of existing U.S. sanctions against the government of North Korea. Mandates that these restrictions remain in effect unless the President certifies to Congress that North Korea has met several conditions, including: (1) ending the illicit transfer of missile or nuclear technology to state sponsors of terrorism; (2) ceasing all support for foreign terrorist organizations and specially designated global terrorists; (3) stopping the counterfeiting of U.S. currency; (4) dismantling Bureau No. 39, which facilitates the laundering of illicit funds; and (5) releasing specified abduction victims, su

### Test bill 66 (10,681 characters), ROUGE-L fine-tuned 0.333, 2-shot Gemini 0.580

**Reference:** First Responders Support Act of 2008 - Amends the Omnibus Crime Control and Safe Streets Act of 1968 to: (1) increase death and disability benefits for public safety officers; and (2) expand educational benefits for dependents of public safety officers killed or disabled in the line of duty.  Amends the Internal Revenue Code to allow a tax credit for up to $7,500 of the medical expenses paid or incurred by a public safety officer or a family member for injuries sustained while such officer was performing an official duty.  Authorizes the Director of the Bureau of Justice Assistance to award grants for mental health services or counseling to public safety officers seriously injured in the lin

**Fine-tuned:** First Responders Support Act of 2008 - Amends the Omnibus Crime Control and Safe Streets Act of 1968 to: (1) increase the maximum amount of disability benefits payable to a public safety officer who becomes disabled as the direct result of a catastrophic injury sustained in the line of duty; (2) require the Director of the Bureau of Justice Assistance to adopt and apply a schedule of levels of disability of public safety officers associated with specific injuries or combinations of injuries; (3) expedite the processing of claims and make disability determinations; (4) authorize appropriations for hiring employees for the Bureau of Justice Assistance and obtaining the technology and equipment

**2-shot Gemini:** First Responders Support Act of 2008 - Amends the Omnibus Crime Control and Safe Streets Act of 1968 to increase death benefits for public safety officers and revise disability benefits for officers sustaining catastrophic injuries in the line of duty. It mandates that the Bureau of Justice Assistance determine benefit eligibility within three months of receiving a claim and authorizes appropriations for staffing and technology to expedite processing.  The Act expands educational assistance for dependents of officers killed or catastrophically injured, increasing monthly payments and establishing annual cost-of-living adjustments. It amends the Internal Revenue Code to provide a refundable t

