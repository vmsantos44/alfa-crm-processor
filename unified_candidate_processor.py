#!/usr/bin/env python3
"""
Unified Candidate Processor with AI Resume Analysis
Downloads resumes, extracts text, sends to OpenAI for tier classification, updates Zoho CRM
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Text extraction libraries
import PyPDF2
from docx import Document

# OpenAI (optional, architecture-specific)
try:
    from openai import OpenAI
    _OPENAI_IMPORT_ERROR = None
except (ImportError, OSError) as err:
    OpenAI = None  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = err

# Zoho integration
from zoho_token_manager import ZohoTokenManager

try:
    import pdfplumber  # Optional; triggers architecture-specific wheels
    _PDFPLUMBER_IMPORT_ERROR = None
except (ImportError, OSError) as err:
    pdfplumber = None  # type: ignore[assignment]
    _PDFPLUMBER_IMPORT_ERROR = err


class UnifiedCandidateProcessor:
    def __init__(self, test_mode=False):
        """
        Initialize the unified processor
        Args:
            test_mode: If True, only process 2 candidates for testing
        """
        self.manager = ZohoTokenManager()
        self.test_mode = test_mode
        self._pdfplumber_warning_logged = False
        self._openai_warning_logged = False

        # Load OpenAI key from environment
        self.openai_client = self._init_openai()
        self.ai_enabled = self.openai_client is not None

        # Directories
        self.download_dir = '/opt/zoho-crm-python/resumes'
        self.results_dir = '/opt/zoho-crm-python/data/processing-results'
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Tag IDs (from config/all_tags.json)
        self.tag_ids = {
            'tier_1': '5827639000001916729',
            'tier_2': '5827639000002158020',
            'tier_3': '5827639000002279535',
            'offshore': '5827639000048896584',
            'automated_ai_review': '5827639000119568099',
            'documents_downloaded': '5827639000119880084',
            'no_resume': '5827639000004776073',
            'no_documents': '5827639000119939036'
        }

        print("🚀 Unified Candidate Processor Initialized")
        print(f"   📁 Resume directory: {self.download_dir}")
        print(f"   🤖 OpenAI: {'Connected' if self.openai_client else 'Not configured'}")
        print(f"   🧪 Test mode: {'ON (max 2 candidates)' if test_mode else 'OFF'}")

    def _init_openai(self) -> Optional["OpenAI"]:
        """Initialize OpenAI client from environment"""
        if OpenAI is None:
            if not self._openai_warning_logged:
                detail = f" ({_OPENAI_IMPORT_ERROR})" if _OPENAI_IMPORT_ERROR else ""
                print(f"⚠️  OpenAI client unavailable{detail}")
                self._openai_warning_logged = True
            return None

        try:
            # Load from ~/.zoho_env
            env_file = os.path.expanduser('~/.zoho_env')
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.strip().split('=', 1)[1]
                            return OpenAI(api_key=api_key)

            print("⚠️  OpenAI API key not found in ~/.zoho_env")
            return None
        except Exception as e:
            print(f"❌ Error initializing OpenAI: {e}")
            return None

    # ========================================================================
    # TEXT EXTRACTION
    # ========================================================================

    def extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from PDF file using pdfplumber with PyPDF2 fallback"""
        # Check file size first
        if os.path.getsize(filepath) == 0:
            print(f"      ⚠️  Empty file (0 bytes): {Path(filepath).name}")
            return ""

        # Try pdfplumber first (better extraction)
        if pdfplumber is not None:
            try:
                text = ""
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text.strip()
            except Exception:
                print(f"      ⚠️  pdfplumber failed, trying PyPDF2...")
        elif not self._pdfplumber_warning_logged:
            detail = f" ({_PDFPLUMBER_IMPORT_ERROR})" if _PDFPLUMBER_IMPORT_ERROR else ""
            print(f"      ⚠️  pdfplumber unavailable{detail}; using PyPDF2 fallback.")
            self._pdfplumber_warning_logged = True

        # Fallback to PyPDF2
        try:
            text = ""
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"      ❌ PDF extraction failed: {e}")
            return ""

    def extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"      ❌ DOCX extraction error: {e}")
            return ""

    def extract_text_from_file(self, filepath: str) -> str:
        """Extract text from file based on extension"""
        ext = Path(filepath).suffix.lower()

        if ext == '.pdf':
            return self.extract_text_from_pdf(filepath)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(filepath)
        elif ext == '.txt':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ""
        else:
            print(f"      ⚠️  Unsupported file type: {ext}")
            return ""

    # ========================================================================
    # AI RESUME ANALYSIS
    # ========================================================================

    def analyze_resume_with_ai(self, resume_text: str, candidate_name: str) -> Dict:
        """
        Send resume to OpenAI for tier classification and analysis
        Returns: dict with tier_level, qualification, analysis, etc.
        """
        if not self.openai_client:
            return {
                'tier_level': 'Unknown',
                'qualification': 'Unable to analyze - OpenAI not configured',
                'error': True
            }

        # Tier classification criteria (from your specification)
        tier_criteria = """
# Interpreter Tier Classification (Alfa Systems)

## Tier 1 Interpreter
- Has at least one year of prior interpreting experience, preferably with Over-the-Phone Interpretation (OPI)
- Has completed at least four hours of official interpreter training
- Ideally holds a certification
- Typically ready for immediate deployment on OPI or VRI accounts

## Tier 2 Interpreter
- Has formal on-site interpretation experience (e.g., hospitals or professional settings)
- Must have at least one year of experience and industry-standard interpretation training
- Does NOT include informal interpreting (e.g., at church, community, or family settings)
- Can be transitioned to OPI through a 2–3 week training program (40 hours total over 5 days)

## Tier 3 Interpreter
- Has no formal or official interpretation experience and no relevant training
- May have bilingual ability but lacks professional interpreting background
- Requires full foundational training before assignment

## Additional Rules:
- Interpreters are categorized as "onshore" (U.S.-based) or "offshore" (outside the U.S.)
- Most offshore interpreters will NOT qualify as Tier 2 unless they have documented on-site interpretation experience
- Simultaneous interpretation alone does NOT qualify for Tier 2 without formal consecutive interpretation experience and training
"""

        prompt = f"""You are an expert interpreter recruiter. Analyze this resume and classify the candidate according to our tier system.

{tier_criteria}

RESUME TEXT:
{resume_text[:4000]}

CANDIDATE NAME: {candidate_name}

Please analyze this resume and provide a JSON response with the following structure:
{{
    "tier_level": "Tier 1" | "Tier 2" | "Tier 3",
    "qualification": "Qualified" | "Not Qualified" | "Needs Review",
    "primary_language": "language name",
    "other_languages": ["lang1", "lang2"],
    "years_of_experience": number or "Unknown",
    "experience_type": "OPI" | "On-site" | "Informal" | "Mixed" | "None",
    "has_training": true | false,
    "training_hours": number or "Unknown",
    "has_certification": true | false,
    "certification_details": "certification name or N/A",
    "service_location": "Onshore" | "Offshore" | "Unknown",
    "location_evidence": "detailed explanation based on phone, education, work",
    "phone_analysis": "US area code or international format details",
    "education_location": "where they studied (city/state/country)",
    "work_location": "where they worked (city/state/country)",
    "current_location_guess": "best assessment from all evidence",
    "remote_experience": true | false,
    "tier_justification": "brief explanation for tier assignment",
    "red_flags": ["flag1", "flag2"] or [],
    "strengths": ["strength1", "strength2"],
    "recommended_action": "brief next step recommendation"
}}

CRITICAL - Location Determination:
- Phone: US area codes (+1, 555-555-5555) = ONSHORE | International (+55, +351, +234) = OFFSHORE
- Education: US schools = ONSHORE | Foreign schools = OFFSHORE
- Work: US companies/cities = ONSHORE | Foreign = OFFSHORE
- Provide specific evidence (e.g., "Phone +1-786-xxx Miami area code, worked at hospital in Florida = ONSHORE")

Be thorough and use evidence from the resume. If information is missing, indicate "Unknown" rather than guessing.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using latest GPT-4 model
                messages=[
                    {"role": "system", "content": "You are an expert interpreter recruiter and resume analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)
            analysis['error'] = False
            return analysis

        except Exception as e:
            print(f"      ❌ OpenAI analysis error: {e}")
            return {
                'tier_level': 'Unknown',
                'qualification': f'Error during analysis: {str(e)}',
                'error': True
            }

    # ========================================================================
    # ZOHO CRM OPERATIONS
    # ========================================================================

    def get_new_candidates_without_ai_review(self) -> List[Dict]:
        """Find all New Candidate leads that haven't been AI reviewed yet"""
        print("\n🔍 Finding New Candidates without AI review...")

        all_candidates = []
        page = 1
        per_page = 200

        while True:
            print(f"   📄 Checking page {page}...")

            endpoint = f'/crm/v2/leads?fields=id,Full_Name,Email,Lead_Status,Tag,Language,Country,Created_Time&per_page={per_page}&page={page}'

            try:
                leads = self.manager.make_api_call(endpoint)

                if not leads.get('data') or len(leads['data']) == 0:
                    break

                for lead in leads['data']:
                    lead_status = lead.get('Lead_Status', '') or ''

                    if 'New Candidate' in lead_status:
                        # Check tags
                        tags = lead.get('Tag', []) or []
                        tag_names = [tag.get('name', '') for tag in tags if isinstance(tag, dict)]

                        # Skip if already reviewed
                        if 'Automated Ai Review' not in tag_names:
                            all_candidates.append({
                                'id': lead['id'],
                                'name': lead.get('Full_Name', 'Unknown'),
                                'email': lead.get('Email', 'N/A'),
                                'language': lead.get('Language', 'Unknown'),
                                'country': lead.get('Country', 'Unknown'),
                                'created': lead.get('Created_Time', 'N/A'),
                                'current_tags': tag_names
                            })

                if len(leads['data']) < per_page:
                    break

                page += 1
                time.sleep(0.2)

            except Exception as e:
                print(f"   ❌ Error on page {page}: {e}")
                break

        print(f"   ✅ Found {len(all_candidates)} candidates needing AI review")

        # Limit in test mode
        if self.test_mode and len(all_candidates) > 2:
            print(f"   🧪 Test mode: limiting to 2 candidates")
            all_candidates = all_candidates[:2]

        return all_candidates

    def download_candidate_attachments(self, lead_id: str, lead_name: str) -> List[str]:
        """Download all attachments for a candidate, return list of file paths"""
        try:
            attachments_response = self.manager.make_api_call(f'/crm/v2/leads/{lead_id}/Attachments')

            if not attachments_response.get('data'):
                return []

            # Create candidate directory
            lead_dir = os.path.join(self.download_dir, self.sanitize_filename(lead_name))
            os.makedirs(lead_dir, exist_ok=True)

            downloaded_files = []

            for attachment in attachments_response['data']:
                filename = attachment.get('File_Name')
                attachment_id = attachment['id']

                # Download attachment
                token = self.manager.get_valid_token()
                url = f"{self.manager.api_domain}/crm/v2/leads/{lead_id}/Attachments/{attachment_id}"

                headers = {'Authorization': f'Zoho-oauthtoken {token}'}
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    filepath = os.path.join(lead_dir, filename)

                    with open(filepath, 'wb') as f:
                        f.write(response.content)

                    downloaded_files.append(filepath)

            return downloaded_files

        except Exception as e:
            print(f"      ❌ Download error: {e}")
            return []

    def add_tags_to_candidate(self, lead_id: str, tag_names: List[str]) -> bool:
        """Add multiple tags to a candidate by tag name"""
        try:
            # Map tag names to tag objects with IDs
            # Zoho requires BOTH name and id for tags to work
            tag_name_to_id = {
                'Tier 1': '5827639000001916729',
                'Tier 2': '5827639000002158020',
                'Tier 3': '5827639000002279535',
                'Offshore': '5827639000048896584',
                'Automated Ai Review': '5827639000119568099',
                'Documents Downloaded': '5827639000119880084',
                'No Resume': '5827639000004776073',
                'No Documents': '5827639000119939036',
                # Language tags
                'Spanish': '5827639000118178599',
                'Portuguese': '5827639000002461322',
                'Brazilian Portuguese': '5827639000002461322',  # Same as Portuguese
                'French': '5827639000002461329',
                'Haitian Creole': '5827639000005703125',
                'French Creole': '5827639000005703125',  # Same as Haitian Creole
                'Swahili': '5827639000018847065',
            }

            # Build tag objects with name and id
            tag_objects = []
            for tag_name in tag_names:
                if tag_name in tag_name_to_id:
                    tag_objects.append({
                        'name': tag_name,
                        'id': tag_name_to_id[tag_name]
                    })
                else:
                    print(f"      ⚠️  Warning: Tag '{tag_name}' not found in system")

            if not tag_objects:
                return False

            data = {
                "data": [{
                    "id": lead_id,
                    "Tag": tag_objects
                }]
            }

            result = self.manager.make_api_call('/crm/v2/leads', 'PUT', data)

            # Check for success
            if isinstance(result.get('data'), list) and len(result['data']) > 0:
                return result['data'][0].get('code') == 'SUCCESS'

            return False

        except Exception as e:
            print(f"      ❌ Tag error: {e}")
            return False

    def add_note_to_candidate(self, lead_id: str, note_content: str, note_title: str = "AI Resume Analysis") -> bool:
        """Add a note to the candidate with AI analysis results"""
        try:
            data = {
                "data": [{
                    "Note_Title": note_title,
                    "Note_Content": note_content,
                    "Parent_Id": lead_id,
                    "se_module": "Leads"
                }]
            }

            result = self.manager.make_api_call('/crm/v2/Notes', 'POST', data)
            return result.get('data', [{}])[0].get('status') == 'success'

        except Exception as e:
            print(f"      ❌ Note error: {e}")
            return False

    def update_lead_status(self, lead_id: str, new_status: str = "Screening") -> Dict[str, any]:
        """
        Update lead status to 'Screening' after AI processing
        Returns: dict with 'success' (bool) and 'message' (str)
        """
        try:
            print(f"   🔄 Attempting to update status to '{new_status}'...")

            data = {
                "data": [{
                    "id": lead_id,
                    "Lead_Status": new_status
                }]
            }

            result = self.manager.make_api_call('/crm/v2/leads', 'PUT', data)

            # Check result
            if isinstance(result.get('data'), list) and len(result['data']) > 0:
                response_data = result['data'][0]

                if response_data.get('code') == 'SUCCESS':
                    print(f"      ✅ Status updated to '{new_status}'")
                    return {'success': True, 'message': f'Status updated to {new_status}'}
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    print(f"      ⚠️  Status update failed: {error_msg}")

                    # Check if it's a Blueprint restriction
                    if 'blueprint' in error_msg.lower() or 'transition' in error_msg.lower():
                        print(f"      ℹ️  Blueprint restriction detected - manual update required")
                        return {'success': False, 'message': 'Blueprint restriction - manual update required', 'blueprint_blocked': True}

                    return {'success': False, 'message': error_msg}

            return {'success': False, 'message': 'No response data'}

        except Exception as e:
            error_msg = str(e)
            print(f"      ❌ Status update error: {error_msg}")

            # Check if Blueprint-related error
            if 'blueprint' in error_msg.lower() or 'transition' in error_msg.lower():
                return {'success': False, 'message': 'Blueprint restriction', 'blueprint_blocked': True}

            return {'success': False, 'message': error_msg}

    def sanitize_filename(self, name: str) -> str:
        """Clean filename for safe directory creation"""
        return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()

    # ========================================================================
    # MAIN PROCESSING WORKFLOW
    # ========================================================================

    def process_candidate(self, candidate: Dict) -> Dict:
        """Process a single candidate: download, analyze, update CRM"""
        lead_id = candidate['id']
        lead_name = candidate['name']

        print(f"\n{'='*70}")
        print(f"👤 Processing: {lead_name}")
        print(f"   📧 {candidate['email']}")
        print(f"   🆔 {lead_id}")
        print(f"   🌍 {candidate['country']} | 🗣️  {candidate['language']}")

        result = {
            'candidate': lead_name,
            'id': lead_id,
            'status': 'processing',
            'downloaded_files': [],
            'analysis': None,
            'tags_added': [],
            'note_added': False
        }

        # Step 1: Download attachments
        print("   📎 Downloading attachments...")
        files = self.download_candidate_attachments(lead_id, lead_name)
        result['downloaded_files'] = files

        if not files:
            print("   ❌ No attachments found")
            # Tag as No Resume
            self.add_tags_to_candidate(lead_id, ['No Resume', 'Automated Ai Review'])
            # Update status to Screening even without resume
            status_update = self.update_lead_status(lead_id, "Screening")
            result['status'] = 'no_resume'
            result['tags_added'] = ['No Resume', 'Automated Ai Review']
            result['status_updated'] = status_update['success']
            result['status_update_message'] = status_update.get('message', 'No resume - moved to Screening')
            if status_update.get('blueprint_blocked'):
                result['blueprint_restriction'] = True
                print("      ℹ️  Manual status change to 'Screening' required")
            return result

        print(f"   ✅ Downloaded {len(files)} file(s)")
        for f in files:
            print(f"      📄 {Path(f).name}")

        # Step 2: Extract text from resume files
        print("   📝 Extracting text from resumes...")
        resume_text = ""

        # First try to extract from resume files
        for filepath in files:
            filename = Path(filepath).name.lower()
            if 'resume' in filename or 'cv' in filename:
                text = self.extract_text_from_file(filepath)
                if text:
                    resume_text += text + "\n\n"
                    print(f"      ✅ Extracted from {Path(filepath).name} ({len(text)} chars)")

        # If resume is empty/missing, try extracting from job application
        if not resume_text or len(resume_text) < 100:
            print("      ⚠️  Resume files empty/missing, trying job application PDF...")
            for filepath in files:
                filename = Path(filepath).name.lower()
                if 'application' in filename or 'jobapplication' in filename:
                    text = self.extract_text_from_file(filepath)
                    if text:
                        resume_text += text + "\n\n"
                        print(f"      ✅ Extracted from {Path(filepath).name} ({len(text)} chars)")

        if not resume_text or len(resume_text) < 100:
            print("   ⚠️  Could not extract sufficient text from any file")
            # Tag as processed but note the issue
            self.add_tags_to_candidate(lead_id, ['Automated Ai Review'])
            self.add_note_to_candidate(lead_id,
                "AI Review attempted but could not extract text from resume files. Files may be empty or corrupted. Manual review required.",
                "AI Review - Extraction Failed")
            # Update status to Screening even with extraction failure
            status_update = self.update_lead_status(lead_id, "Screening")
            result['status'] = 'extraction_failed'
            result['tags_added'] = ['Automated Ai Review']
            result['note_added'] = True
            result['status_updated'] = status_update['success']
            result['status_update_message'] = status_update.get('message', 'Extraction failed - moved to Screening')
            if status_update.get('blueprint_blocked'):
                result['blueprint_restriction'] = True
                print("      ℹ️  Manual status change to 'Screening' required")
            return result

        if not self.ai_enabled:
            print("   ⚠️  OpenAI unavailable; skipping AI analysis and flagging for manual review.")
            fallback_tags = ['Documents Downloaded', 'Automated Ai Review']
            tags_applied: List[str] = []
            if self.add_tags_to_candidate(lead_id, fallback_tags):
                tags_applied = fallback_tags
                print(f"      ✅ Added fallback tags: {', '.join(tags_applied)}")
            else:
                print("      ❌ Failed to add fallback tags")

            note_added = self.add_note_to_candidate(
                lead_id,
                "AI analysis skipped because the OpenAI client is not available. "
                "Resume files were downloaded successfully. Please complete tier assignment manually.",
                "AI Review - Manual Review Required"
            )
            if note_added:
                print("      ✅ Added manual review note")
            else:
                print("      ❌ Failed to add manual review note")

            # Update status to Screening even without AI analysis
            status_update = self.update_lead_status(lead_id, "Screening")
            result['status'] = 'ai_skipped'
            result['analysis'] = {
                'tier_level': 'Unknown',
                'qualification': 'AI analysis skipped - OpenAI unavailable',
                'error': True
            }
            result['tags_added'] = tags_applied
            result['note_added'] = note_added
            result['status_updated'] = status_update['success']
            result['status_update_message'] = status_update.get('message', 'AI skipped - moved to Screening')
            if status_update.get('blueprint_blocked'):
                result['blueprint_restriction'] = True
                print("      ℹ️  Manual status change to 'Screening' required")
            return result

        # Step 3: AI Analysis
        print("   🤖 Analyzing resume with OpenAI...")
        analysis = self.analyze_resume_with_ai(resume_text, lead_name)
        result['analysis'] = analysis

        if analysis.get('error'):
            print(f"   ❌ Analysis failed: {analysis.get('qualification')}")
            result['status'] = 'analysis_failed'
            return result

        # Print analysis results
        print(f"   📊 Analysis Results:")
        print(f"      🎯 Tier: {analysis.get('tier_level')}")
        print(f"      ✅ Qualification: {analysis.get('qualification')}")
        print(f"      🗣️  Primary Language: {analysis.get('primary_language')}")
        print(f"      📍 Location: {analysis.get('service_location')}")
        print(f"      💼 Experience: {analysis.get('years_of_experience')} years ({analysis.get('experience_type')})")
        print(f"      🎓 Training: {analysis.get('training_hours')} hours" if analysis.get('has_training') else "      🎓 Training: None")
        print(f"      📜 Certification: {analysis.get('certification_details')}")

        # Step 4: Determine tags to add
        tag_names = []

        # Tier tag
        tier = analysis.get('tier_level', '').lower()
        if 'tier 1' in tier:
            tag_names.append('Tier 1')
        elif 'tier 2' in tier:
            tag_names.append('Tier 2')
        elif 'tier 3' in tier:
            tag_names.append('Tier 3')

        # Location tag
        location = analysis.get('service_location', '').lower()
        if 'offshore' in location:
            tag_names.append('Offshore')

        # Language tag (based on CRM Language field, not AI analysis)
        crm_language = candidate.get('language', '')
        if crm_language and crm_language.lower() != 'english':
            # Check if we have a tag for this language
            language_tag_map = {
                'Spanish': 'Spanish',
                'Portuguese': 'Portuguese',
                'Brazilian Portuguese': 'Portuguese',
                'French': 'French',
                'Haitian Creole': 'Haitian Creole',
                'French Creole': 'Haitian Creole',
                'Swahili': 'Swahili',
                'Kinyarwanda': 'Kinyarwanda',
                'Somali': 'Somali'
            }

            tag_to_add = language_tag_map.get(crm_language)
            if tag_to_add and tag_to_add not in tag_names:
                tag_names.append(tag_to_add)

        # Always add these
        tag_names.extend(['Automated Ai Review', 'Documents Downloaded'])

        # Step 5: Add tags
        print(f"   🏷️  Adding tags: {', '.join(tag_names)}")
        if self.add_tags_to_candidate(lead_id, tag_names):
            print("      ✅ Tags added successfully")
            result['tags_added'] = tag_names
        else:
            print("      ❌ Failed to add tags")

        # Step 6: Add analysis note
        # Determine if this is an edge case requiring manual review
        tier = analysis.get('tier_level', '')
        years_exp = analysis.get('years_of_experience', 0)
        review_note = ""

        # Flag edge cases for manual review
        # Convert years_exp to float for comparison, handle 'Unknown' case
        try:
            years_exp_num = float(years_exp) if years_exp != 'Unknown' else None
        except (ValueError, TypeError):
            years_exp_num = None

        if tier == 'Tier 1' and (years_exp_num is None or years_exp_num < 1):
            review_note = "\n⚠️ EDGE CASE: Less than 1 year experience - May need manual review to confirm Tier 1 classification."
        elif tier == 'Tier 2' and not analysis.get('has_training'):
            review_note = "\n⚠️ EDGE CASE: No documented training - Verify on-site experience qualifies for Tier 2."
        elif tier == 'Tier 3' and years_exp_num is not None and years_exp_num > 0:
            review_note = "\n⚠️ EDGE CASE: Has some experience but classified Tier 3 - Resume may be incomplete. Consider interview to clarify."

        note_content = f"""AI Resume Analysis Results
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TIER CLASSIFICATION: {analysis.get('tier_level')}
QUALIFICATION: {analysis.get('qualification')}
{review_note}

JUSTIFICATION:
{analysis.get('tier_justification', 'N/A')}

LANGUAGES:
- Primary: {analysis.get('primary_language')}
- Other: {', '.join(analysis.get('other_languages', []))}

EXPERIENCE:
- Years: {analysis.get('years_of_experience')}
- Type: {analysis.get('experience_type')}
- Remote Experience: {'Yes' if analysis.get('remote_experience') else 'No'}

TRAINING & CERTIFICATION:
- Training Hours: {analysis.get('training_hours')}
- Certification: {analysis.get('certification_details')}

LOCATION ANALYSIS:
- Service Location: {analysis.get('service_location')}
- Phone Analysis: {analysis.get('phone_analysis', 'Not analyzed')}
- Education Location: {analysis.get('education_location', 'Not specified')}
- Work Location: {analysis.get('work_location', 'Not specified')}
- Current Location (Best Guess): {analysis.get('current_location_guess', 'Unknown')}
- Evidence Summary: {analysis.get('location_evidence', 'N/A')}

STRENGTHS:
{chr(10).join(['- ' + s for s in analysis.get('strengths', [])])}

RED FLAGS:
{chr(10).join(['- ' + f for f in analysis.get('red_flags', [])]) if analysis.get('red_flags') else 'None identified'}

RECOMMENDED ACTION:
{analysis.get('recommended_action', 'N/A')}

NEXT STATUS:
✅ Ready for Screening - Documents downloaded and AI analysis complete.
Status will be automatically updated to "Screening" if Blueprint permits.

---
This analysis was performed automatically using AI. Please review for accuracy.
Resume may not reflect all qualifications - consider during interview for final tier determination.
"""

        print("   📝 Adding analysis note to CRM...")
        if self.add_note_to_candidate(lead_id, note_content):
            print("      ✅ Note added successfully")
            result['note_added'] = True
        else:
            print("      ❌ Failed to add note")

        # Attempt to update status to Screening
        status_update = self.update_lead_status(lead_id, "Screening")
        result['status_updated'] = status_update['success']
        result['status_update_message'] = status_update['message']

        if status_update.get('blueprint_blocked'):
            result['blueprint_restriction'] = True
            print("      ℹ️  Manual status change to 'Screening' required")

        result['status'] = 'completed'
        print(f"   ✅ Processing complete for {lead_name}")

        return result

    def process_all_candidates(self) -> Dict:
        """Main workflow: find and process all candidates"""
        print("\n" + "="*70)
        print("🚀 UNIFIED CANDIDATE PROCESSOR - AI RESUME ANALYSIS")
        print("="*70)

        start_time = datetime.now()

        # Get candidates to process
        candidates = self.get_new_candidates_without_ai_review()

        if not candidates:
            print("\n✅ No candidates need processing!")
            return {'total': 0, 'results': []}

        print(f"\n📊 Processing {len(candidates)} candidate(s)...")

        # Process each candidate
        results = []
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}/{len(candidates)}]")
            result = self.process_candidate(candidate)
            results.append(result)

            # Delay between candidates to respect rate limits
            if i < len(candidates):
                time.sleep(3)

        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            'total_processed': len(results),
            'successful': sum(1 for r in results if r['status'] == 'completed'),
            'no_resume': sum(1 for r in results if r['status'] == 'no_resume'),
            'manual_review_required': sum(1 for r in results if r['status'] == 'ai_skipped'),
            'failed': sum(1 for r in results if r['status'] in ['extraction_failed', 'analysis_failed']),
            'status_auto_updated': sum(1 for r in results if r.get('status_updated')),
            'status_manual_required': sum(1 for r in results if r.get('blueprint_restriction')),
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

        # Save detailed results
        results_file = os.path.join(
            self.results_dir,
            f"ai_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"   ✅ Successfully processed: {summary['successful']}")
        print(f"   ⚠️  No resume found: {summary['no_resume']}")
        print(f"   📝 Manual review required: {summary['manual_review_required']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   🔄 Status auto-updated to Screening: {summary['status_auto_updated']}")
        if summary['status_manual_required'] > 0:
            print(f"   👤 Manual status update required: {summary['status_manual_required']}")
        print(f"   ⏱️  Duration: {duration:.1f} seconds")
        print(f"   📄 Detailed results: {results_file}")

        # Show tier distribution
        tier_counts = {}
        for r in results:
            if r.get('analysis') and not r['analysis'].get('error'):
                tier = r['analysis'].get('tier_level', 'Unknown')
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        if tier_counts:
            print("\n   🎯 Tier Distribution:")
            for tier, count in sorted(tier_counts.items()):
                print(f"      {tier}: {count}")

        print("\n✅ Processing complete!")

        return summary


def main():
    """Main entry point"""
    import sys

    # Check for test mode flag
    args = sys.argv[1:]
    test_mode = '--test' in args or '-t' in args
    allow_missing_openai = '--allow-missing-openai' in args

    processor = UnifiedCandidateProcessor(test_mode=test_mode)

    if not processor.openai_client:
        if not allow_missing_openai:
            print("\n❌ Cannot proceed without OpenAI API key configured")
            print("   Please ensure OPENAI_API_KEY is set in ~/.zoho_env or rerun with --allow-missing-openai")
            return
        print("\n⚠️  OpenAI API key not configured; continuing in manual review mode.")

    # Run processing
    processor.process_all_candidates()


if __name__ == "__main__":
    main()
