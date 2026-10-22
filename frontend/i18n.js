/* ================================================================
   Skin-Glow — Thai / English switch for the user pages
   (index.html, login.html, reset_password.html).

   The pages are written in Thai. When English is chosen this script
   translates the page in place: every text node and the placeholder /
   title / aria-label / alt attributes, including content the page adds
   later (a MutationObserver), plus alert()/confirm() messages and the
   Thai messages the PHP API returns. In Thai mode the few English labels
   (menu, section titles, concern names such as "acne") are shown in Thai.

     I18N.lang            'th' | 'en' (saved in localStorage)
     I18N.locale          'th-TH' | 'en-GB' for toLocaleDateString
     I18N.setLang('en')   saves the choice and reloads the page
     I18N.tr('ข้อความ')   English for a Thai string in English mode
     I18N.articles        English Daily Insights articles, by id

   Lookup order for a piece of text: exact match in DICT -> PATTERNS
   (text with numbers / dates in it) -> FRAGMENTS (zone and concern
   names inside longer text). Text with no match stays Thai.
================================================================= */
(function () {
  'use strict';

  const STORE_KEY = 'skinglow_lang';
  let lang = 'th';
  try { lang = localStorage.getItem(STORE_KEY) === 'en' ? 'en' : 'th'; } catch (e) { /* storage blocked */ }

  // ---------------------------------------------------------------- exact strings
  const DICT = {
    // ---- page titles
    'Skin-Glow — วิเคราะห์ผิวหน้าด้วย AI': 'Skin-Glow — AI Facial Skin Analysis',
    'เข้าสู่ระบบ — Skin-Glow': 'Log in — Skin-Glow',
    'ตั้งรหัสผ่านใหม่ — Skin-Glow': 'Reset password — Skin-Glow',

    // ---- header / navigation
    'เปิดเมนู': 'Open menu',
    'โปรไฟล์ของฉัน': 'My profile',
    'ออกจากระบบ': 'Log out',

    // ---- Analysis tab
    'รู้จักผิวคุณ': 'Know your skin',
    'ก่อนเลือกสกินแคร์': 'before choosing skincare',
    'อัปโหลดรูปหรือถ่ายภาพใบหน้า ระบบ AI จะตรวจจับปัญหาผิว — สิว จุดด่างดำ ถุงใต้ตา รอยแดง ความมัน และริ้วรอย — พร้อมแนะนำผลิตภัณฑ์ที่ตรงจุดสำหรับคุณ':
      'Upload or take a photo of your face. The AI detects skin concerns — acne, dark spots, eye bags, redness, oiliness and wrinkles — and suggests products that target them.',
    'ปัญหาผิวที่ตรวจจับได้': 'skin concerns detected',
    'โซนใบหน้าที่วิเคราะห์': 'face zones analysed',
    'โมเดล AI ทำงานร่วมกัน': 'AI models working together',
    'กำลังวิเคราะห์ผิว...': 'Analysing your skin...',
    'กำลังตรวจแสง…': 'Checking light…',
    'จัดใบหน้าให้อยู่ในกรอบ มองตรง': 'Keep your face inside the frame and look straight ahead',
    'ลากรูปมาวาง หรือคลิกเพื่ออัปโหลด': 'Drag a photo here, or click to upload',
    'เลือกรูปภาพ': 'Choose photo',
    'เปิดกล้อง': 'Open camera',
    'ถ่ายภาพ': 'Take photo',
    'ยกเลิก': 'Cancel',
    'ถ่ายใหม่': 'Retake',
    'เลือกรูปอื่น': 'Choose another photo',
    'ภาพตัวอย่างใบหน้า': 'Face preview',
    'แสงน้อยเกินไป ลองหันหน้าเข้าหาแสง': 'Too dark — turn towards the light',
    'แสงจ้าเกินไป ลองหลบแสงที่ส่องตรง': 'Too bright — move out of direct light',
    'แสงพอดี ✓': 'Lighting OK ✓',
    'กล้องยังไม่พร้อม กรุณารอสักครู่แล้วลองอีกครั้ง': 'The camera is not ready yet. Please wait a moment and try again.',
    'ไม่สามารถถ่ายภาพได้ กรุณาลองใหม่อีกครั้ง': 'Could not take the photo. Please try again.',

    // ---- results
    'ผลการวิเคราะห์ผิว': 'Skin analysis result',
    'ถ่ายภาพใหม่': 'New photo',
    'คะแนนสุขภาพผิว': 'Skin health score',
    'ยิ่งคะแนนสูง ผิวยิ่งอยู่ในสภาพดี': 'The higher the score, the healthier the skin',
    'ผลวิเคราะห์แยกตามส่วนของใบหน้า': 'Results by face zone',
    'ตัวเลขมุมขวาคือคะแนนของโซน (100 = ไม่พบปัญหา) · % คือความมั่นใจของโมเดล':
      'The number at the top right is the zone score (100 = no concern found) · % is the model\'s confidence',
    'วิเคราะห์ด้วยโมเดลจำแนกรายโซน · % คือความมั่นใจของโมเดล': 'Analysed by the per-zone classifier · % is the model\'s confidence',
    'วิเคราะห์จากจุดที่ตรวจพบในแต่ละโซน · % คือความมั่นใจของโมเดล': 'Analysed from the spots detected in each zone · % is the model\'s confidence',
    'ผลิตภัณฑ์แนะนำสำหรับคุณ': 'Recommended products for you',
    'คัดเลือกตามปัญหาผิวที่ตรวจพบ': 'Picked for the concerns that were found',
    'ผลการวิเคราะห์พร้อมกรอบตรวจจับ': 'Analysis result with detection boxes',
    'ไม่พบปัญหาชัดเจน': 'No clear concern',
    'ไม่พบปัญหาผิวที่ชัดเจนในทุกโซน': 'No clear skin concern in any zone',
    'ไม่พบปัญหาผิวที่ชัดเจน': 'No clear skin concern found',
    'ผิวคุณดูใสสะอาด ไม่พบปัญหาที่ชัดเจน 🎉': 'Your skin looks clear — no obvious concerns 🎉',
    'วิเคราะห์เสร็จสมบูรณ์': 'Analysis complete',
    'สแกนของคุณ': 'Your scan',

    // ---- Daily Insights
    'เกร็ดความรู้เชิงลึกเรื่องผิวและสกินแคร์ คัดสรรให้เข้ากับปัญหาผิวที่ตรวจพบล่าสุดของคุณ อัปเดตใหม่ทุกวัน':
      'In-depth skin and skincare tips, matched to the concerns found in your latest scan. Updated daily.',
    'ℹ️ เนื้อหาเพื่อการศึกษาทั่วไปเท่านั้น ไม่ใช่คำวินิจฉัยหรือคำแนะนำทางการแพทย์ หากมีปัญหาผิวรุนแรงหรือเรื้อรัง ควรปรึกษาแพทย์ผิวหนัง':
      'ℹ️ General educational content only — not a diagnosis or medical advice. For severe or long-lasting skin problems, see a dermatologist.',
    'แนะนำสำหรับผิวคุณวันนี้': 'Picked for your skin today',
    'แนะนำสำหรับผิวคุณ': 'For your skin',
    'คัดสรรวันนี้': 'Today\'s pick',
    'ทั้งหมด': 'All',

    // ---- Insight tab
    'ปริมาณปัญหาผิวแต่ละประเภทในแต่ละวัน เพื่อดูว่าผิวคุณเปลี่ยนแปลงไปมากแค่ไหนเมื่อเทียบกับช่วงก่อนหน้า':
      'How many of each skin concern were found each day, so you can see how much your skin has changed.',
    'ยังไม่มีประวัติการวิเคราะห์ของคุณ — กราฟด้านล่างเป็นข้อมูลตัวอย่างเพื่อให้เห็นภาพ ลองสแกนผิวครั้งแรกที่แท็บ "วิเคราะห์ผิว" เพื่อเริ่มเก็บประวัติจริงของคุณเอง':
      'You have no analysis history yet — the chart below shows example data. Do your first scan on the Analysis tab to start building your own history.',
    'ประวัติการวิเคราะห์ของคุณ': 'Your analysis history',
    'ยังไม่มีประวัติการวิเคราะห์': 'No analysis history yet',
    'คะแนนผิว (เส้น)': 'Skin score (line)',
    'ยังไม่มีข้อมูลเพียงพอสำหรับแสดงกราฟ ลองวิเคราะห์ผิวของคุณก่อน': 'Not enough data for the chart yet. Analyse your skin first.',
    'ลบผลวิเคราะห์นี้': 'Delete this result',
    'ภาพผลการวิเคราะห์': 'Analysis image',
    'ลบผลวิเคราะห์และรูปภาพนี้ออกจากประวัติ? ลบแล้วจะกู้คืนไม่ได้': 'Delete this result and its photo from your history? This cannot be undone.',
    'เกิดข้อผิดพลาด': 'Something went wrong',
    'ปิด': 'Close',
    'ปิดรูปภาพ': 'Close image',
    'ภาพผลการวิเคราะห์ผิว': 'Skin analysis image',
    'ย่อรูป': 'Zoom out',
    'ขยายรูป': 'Zoom in',
    'คืนค่าขนาดรูป': 'Reset zoom',

    // ---- Compare
    'เลือกผลวิเคราะห์ 2 ครั้งใดก็ได้มาเปรียบเทียบ เพื่อดูว่าผิวดีขึ้นหรือแย่ลง (ค่าเริ่มต้นคือครั้งล่าสุดกับครั้งก่อนหน้า)':
      'Pick any two results to compare and see whether your skin got better or worse (default: latest vs the one before).',
    'ครั้งก่อน': 'Before',
    'ครั้งหลัง': 'After',
    'ล่าสุด': 'Latest',
    'ล่าสุด กับ ครั้งก่อนหน้า': 'Latest vs previous',
    'ล่าสุด กับ ครั้งแรก': 'Latest vs first',
    'เลือกผลวิเคราะห์ 2 ครั้งที่ต่างกันเพื่อเปรียบเทียบ': 'Choose two different results to compare',
    'คงที่': 'same',
    'วันเดียวกัน': 'same day',
    'คะแนนผิว': 'Skin score',
    'ผลวิเคราะห์ครั้งก่อน': 'Earlier result',
    'ผลวิเคราะห์ครั้งหลัง': 'Later result',

    // ---- profile modal
    'ข้อมูลส่วนตัว': 'Personal details',
    'ชื่อ-นามสกุล': 'Full name',
    'อีเมล': 'Email',
    'อีเมลใช้เข้าสู่ระบบ เปลี่ยนไม่ได้': 'Your email is used to log in and cannot be changed',
    'ประเภทผิว': 'Skin type',
    'ไม่ระบุ': 'Not specified',
    'ผิวธรรมดา (normal)': 'Normal',
    'ผิวมัน (oily)': 'Oily',
    'ผิวแห้ง (dry)': 'Dry',
    'ผิวผสม (combination)': 'Combination',
    'ผิวแพ้ง่าย (sensitive)': 'Sensitive',
    'บันทึกข้อมูล': 'Save',
    'เปลี่ยนรหัสผ่าน': 'Change password',
    'รหัสผ่านปัจจุบัน': 'Current password',
    'รหัสผ่านใหม่': 'New password',
    'ยืนยันรหัสผ่านใหม่': 'Confirm new password',
    'ลบบัญชี': 'Delete account',
    'ลบบัญชี ผลวิเคราะห์ และรูปภาพทั้งหมดของคุณอย่างถาวร กู้คืนไม่ได้': 'Permanently delete your account, results and photos. This cannot be undone.',
    'ยืนยันด้วยรหัสผ่าน': 'Confirm with your password',
    'ลบบัญชีของฉัน': 'Delete my account',
    'กรุณาเข้าสู่ระบบใหม่': 'Please log in again',
    'กรุณากรอกชื่อ-นามสกุล': 'Please enter your full name',
    'กรุณากรอกรหัสผ่านปัจจุบัน': 'Please enter your current password',
    'ยืนยันรหัสผ่านใหม่ไม่ตรงกัน': 'The new passwords do not match',
    'กรุณากรอกรหัสผ่านเพื่อยืนยัน': 'Please enter your password to confirm',
    'ยืนยันลบบัญชี? ผลวิเคราะห์และรูปภาพทั้งหมดจะถูกลบถาวร และกู้คืนไม่ได้': 'Delete your account? All results and photos will be deleted permanently and cannot be recovered.',

    // ---- login / register / forgot / reset
    'ให้ทุกวันของผิวคุณ เปล่งประกายความสดชื่น': 'Let your skin glow, every day',
    'วิเคราะห์ผิว ติดตามความเปลี่ยนแปลง และรับคำแนะนำที่ออกแบบมาสำหรับผิวคุณโดยเฉพาะ': 'Analyse your skin, track how it changes and get advice made for your skin.',
    '6 ปัญหาผิว': '6 skin concerns',
    'ยินดีต้อนรับกลับ': 'Welcome back',
    'เข้าสู่ระบบ': 'Log in',
    'เข้าสู่บัญชีของคุณเพื่อดูผลวิเคราะห์ผิวและคำแนะนำล่าสุด': 'Log in to see your latest skin results and advice.',
    'รหัสผ่าน': 'Password',
    'จำฉันไว้': 'Remember me',
    'ลืมรหัสผ่าน?': 'Forgot password?',
    'สมัครสมาชิกใหม่': 'Create an account',
    'เริ่มต้นใช้งาน': 'Get started',
    'สมัครสมาชิก': 'Sign up',
    'สร้างบัญชีเพื่อเริ่มวิเคราะห์ผิวและติดตามความเปลี่ยนแปลงของคุณ': 'Create an account to start analysing your skin and tracking its changes.',
    'มีบัญชีอยู่แล้ว? เข้าสู่ระบบ': 'Already have an account? Log in',
    'กู้คืนบัญชี': 'Account recovery',
    'ลืมรหัสผ่าน': 'Forgot password',
    'กรอกอีเมลที่ใช้สมัคร เราจะส่งลิงก์สำหรับตั้งรหัสผ่านใหม่ไปให้': 'Enter the email you signed up with and we will send you a link to set a new password.',
    'ส่งลิงก์ตั้งรหัสผ่านใหม่': 'Send reset link',
    'กลับไปหน้าเข้าสู่ระบบ': 'Back to log in',
    'การใช้งานถือว่าคุณยอมรับ': 'By using Skin-Glow you agree to the',
    'ข้อกำหนดการใช้งาน': 'Terms of Use',
    'และ': 'and',
    'นโยบายความเป็นส่วนตัว': 'Privacy Policy',
    'แสดงรหัสผ่าน': 'Show password',
    'ซ่อนรหัสผ่าน': 'Hide password',
    'ชื่อของคุณ': 'Your name',
    'อย่างน้อย 8 ตัวอักษร': 'At least 8 characters',
    'ตั้งรหัสผ่านใหม่': 'Set a new password',
    'กรอกรหัสผ่านใหม่ที่ต้องการใช้ (อย่างน้อย 8 ตัวอักษร)': 'Enter the new password you want to use (at least 8 characters)',
    'บันทึกรหัสผ่านใหม่': 'Save new password',
    'พิมพ์รหัสผ่านใหม่อีกครั้ง': 'Type the new password again',
    'ลบบัญชีและข้อมูลทั้งหมดของคุณเรียบร้อยแล้ว': 'Your account and all its data have been deleted.',
    'ตั้งรหัสผ่านใหม่เรียบร้อยแล้ว เข้าสู่ระบบด้วยรหัสผ่านใหม่ได้เลย': 'Your password has been reset. You can log in with the new password.',
    'กำลังส่ง...': 'Sending...',
    'ส่งคำขอไม่สำเร็จ': 'Request failed',
    'กำลังเข้าสู่ระบบ...': 'Logging in...',
    'เข้าสู่ระบบไม่สำเร็จ': 'Login failed',
    'กำลังสมัครสมาชิก...': 'Signing up...',
    'สมัครสมาชิกไม่สำเร็จ': 'Sign-up failed',
    'ลิงก์ไม่ถูกต้อง กรุณาเปิดลิงก์จากอีเมลอีกครั้ง หรือขอลิงก์ใหม่ที่หน้าเข้าสู่ระบบ (กด "ลืมรหัสผ่าน?")':
      'This link is not valid. Open the link from the email again, or request a new one on the login page ("Forgot password?").',
    'ลิงก์นี้หมดอายุหรือถูกใช้ไปแล้ว': 'This link has expired or has already been used',
    'รหัสผ่านทั้งสองช่องไม่ตรงกัน': 'The two passwords do not match',
    'กำลังบันทึก...': 'Saving...',
    'ตั้งรหัสผ่านใหม่ไม่สำเร็จ': 'Could not reset the password',
    'เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ': 'An unknown error occurred',

    // ---- messages returned by the PHP API
    'กรุณาเข้าสู่ระบบก่อนใช้งาน': 'Please log in first',
    'ต้องใช้สิทธิ์ผู้ดูแลระบบสำหรับส่วนนี้': 'Administrator access is required',
    'กรุณากรอกอีเมลให้ถูกต้อง': 'Please enter a valid email address',
    'หากอีเมลนี้มีบัญชีอยู่ในระบบ เราได้ส่งลิงก์สำหรับตั้งรหัสผ่านใหม่ไปแล้ว กรุณาตรวจสอบกล่องจดหมาย (รวมถึงโฟลเดอร์สแปม)':
      'If this email has an account, we have sent a password reset link. Please check your inbox (and spam folder).',
    'ลิงก์นี้ไม่ถูกต้อง หมดอายุ หรือถูกใช้ไปแล้ว กรุณาขอลิงก์ใหม่อีกครั้ง': 'This link is invalid, expired or already used. Please request a new link.',
    'รหัสผ่านต้องมีอย่างน้อย 8 ตัวอักษร': 'The password must be at least 8 characters',
    'เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง': 'Something went wrong. Please try again.',
    'ตั้งรหัสผ่านใหม่เรียบร้อยแล้ว กรุณาเข้าสู่ระบบด้วยรหัสผ่านใหม่': 'Your password has been reset. Please log in with the new password.',
    'รหัสผ่านใหม่ต้องมีอย่างน้อย 8 ตัวอักษร': 'The new password must be at least 8 characters',
    'รหัสผ่านปัจจุบันไม่ถูกต้อง': 'The current password is incorrect',
    'รหัสผ่านใหม่ต้องไม่ซ้ำกับรหัสผ่านเดิม': 'The new password must be different from the old one',
    'เปลี่ยนรหัสผ่านเรียบร้อยแล้ว': 'Password changed',
    'ไม่พบผลวิเคราะห์นี้ในประวัติของคุณ': 'This result was not found in your history',
    'ลบผลวิเคราะห์เรียบร้อยแล้ว': 'Result deleted',
    'ไม่พบใบหน้าในภาพ กรุณาถ่ายภาพใบหน้าให้ชัดเจนแล้วลองใหม่อีกครั้ง': 'No face was found in the photo. Please take a clear photo of your face and try again.',
    'ไม่พบบัญชีผู้ใช้': 'Account not found',
    'กรุณากรอกชื่อ-นามสกุล (ไม่เกิน 150 ตัวอักษร)': 'Please enter your full name (up to 150 characters)',
    'ประเภทผิวไม่ถูกต้อง': 'Invalid skin type',
    'บันทึกข้อมูลเรียบร้อยแล้ว': 'Saved',
    'รหัสผ่านไม่ถูกต้อง': 'Incorrect password',
    'บัญชีผู้ดูแลระบบลบด้วยตนเองไม่ได้': 'An administrator account cannot delete itself',
    'ลบบัญชีและข้อมูลทั้งหมดเรียบร้อยแล้ว': 'Your account and all its data have been deleted',
  };

  // ------------------------------------------------- text with numbers / names in it
  // [regex on the whole (whitespace-normalised) text, replacement]
  const PATTERNS = [
    [/^ไม่สามารถเข้าถึงกล้องได้: (.*)$/, (m, x) => 'Cannot access the camera: ' + sub(x)],
    [/^สมาชิกตั้งแต่ (.*)$/, 'Member since $1'],
    [/^โหลดข้อมูลไม่สำเร็จ: (.*)$/, (m, x) => 'Could not load data: ' + sub(x)],
    [/^ไม่สามารถวิเคราะห์รูปภาพได้: (.*)$/, (m, x) => 'Could not analyse the photo: ' + sub(x)],
    [/^ลบไม่สำเร็จ: (.*)$/, (m, x) => 'Delete failed: ' + sub(x)],
    [/^(API error \d+): (.*)$/, (m, a, x) => a + ': ' + sub(x)],
    [/^เซิร์ฟเวอร์ตอบกลับไม่ถูกต้อง \(HTTP (\d+)\)$/, 'Invalid server response (HTTP $1)'],
    [/^ต้องมีผลวิเคราะห์อย่างน้อย 2 ครั้งจึงจะเปรียบเทียบได้ \(ตอนนี้มี (\d+) ครั้ง\) ลองสแกนผิวของคุณอีกครั้ง$/,
      'You need at least 2 results to compare (you have $1). Scan your skin again.'],
    [/^ห่างกัน (\d+) วัน$/, (m, n) => `${n} day${n === '1' ? '' : 's'} apart`],
    [/^(\d+) นาที$/, '$1 min'],
    [/^(.+) อ่าน · แตะเพื่อดูเนื้อหาเต็ม$/, '$1 read · tap to read more'],
    [/^(.*) · คะแนน (\S+)$/, '$1 · score $2'],
    [/^(.*) • คะแนนผิว (\S+)$/, '$1 • skin score $2'],
    [/^(.*) \(ตัวอย่าง\)$/, '$1 (example)'],
    [/^ลบผลวิเคราะห์ (.*)$/, 'Delete result $1'],
    [/^ดูภาพผลการวิเคราะห์ (.*)$/, 'View analysis image $1'],
    [/^(.*) · ไม่ได้ประเมิน(.+) เพราะมีผมหรือสิ่งอื่นบังอยู่ หากต้องการผลของโซนนี้ให้เปิดหน้าผากแล้วถ่ายใหม่$/,
      (m, before, zones) => `${tr(before)} · ${frag(zones)} not assessed because hair or something else covers it — uncover it and retake the photo to get this zone`],
  ];

  // -------------------------------------- names that appear inside longer text
  const FRAGMENTS = {
    'หน้าผาก': 'forehead', 'แก้มซ้าย': 'left cheek', 'แก้มขวา': 'right cheek',
    'จมูก': 'nose', 'ใต้ตา': 'under-eye', 'คาง': 'chin',
    'สิว': 'acne', 'จุดด่างดำ': 'dark spots', 'ถุงใต้ตา': 'eye bags',
    'ความมัน': 'oiliness', 'รอยแดง': 'redness', 'ริ้วรอย': 'wrinkles',
    'กรุณาลองใหม่': 'please try again',
    'เกิดข้อผิดพลาด': 'Something went wrong',
    'หากเพิ่ง deploy หรือไม่มีคนใช้งานนานเกิน 15 นาที AI service บน Render free tier อาจกำลัง sleep อยู่ — รอ 30-60 วินาทีแล้วลองใหม่อีกครั้ง':
      'If the service was just deployed or nobody used it for over 15 minutes, the AI service on Render\'s free tier may be asleep — wait 30-60 seconds and try again.',
    '(กด "ลืมรหัสผ่าน?" ที่หน้าเข้าสู่ระบบ)': '(use "Forgot password?" on the login page)',
  };
  // longer dictionary sentences also work as fragments, so a message glued
  // from two known sentences (API message + our hint) is still translated
  Object.keys(DICT).forEach(k => { if (k.length >= 10 && !(k in FRAGMENTS)) FRAGMENTS[k] = DICT[k]; });
  const FRAG_KEYS = Object.keys(FRAGMENTS).sort((a, b) => b.length - a.length);
  const FRAG_RE = new RegExp(FRAG_KEYS.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');

  const HAS_THAI = /[฀-๿]/;
  const norm = s => s.replace(/\s+/g, ' ').trim();

  // ------------------------------------------------ Thai mode: the English labels
  // (menu, section titles, concern names) shown in Thai when TH is chosen
  const ISSUE_TH = {
    acne: 'สิว', black_spot: 'จุดด่างดำ', eyebag: 'ถุงใต้ตา', redness: 'รอยแดง',
    oiliness: 'ความมัน', wrinkle: 'ริ้วรอย', general: 'ทั่วไป',
  };
  const DICT_TH = Object.assign({
    'Analysis': 'วิเคราะห์ผิว',
    'Daily Insights': 'สาระน่ารู้รายวัน',
    'Insight': 'ติดตามผลการวิเคราะห์',
    'Compare My Skin': 'เปรียบเทียบผิวของคุณ',
    'Admin Panel': 'ผู้ดูแลระบบ',
    'AI Skin Analysis': 'วิเคราะห์ผิวด้วย AI',
    'Daily Personalised Insights': 'สาระน่ารู้รายวันสำหรับคุณ',
    '© 2026 Skin-Glow — AI Skin Analysis & Skincare Recommendation': '© 2026 Skin-Glow — วิเคราะห์ผิวด้วย AI และแนะนำสกินแคร์',
  }, ISSUE_TH);
  const TH_PATTERNS = [
    [/^([a-z_]+) × (\d+)$/, (m, k, n) => (ISSUE_TH[k] ? `${ISSUE_TH[k]} × ${n}` : m)],
  ];
  function trTh(s) {
    const key = norm(s);
    if (Object.prototype.hasOwnProperty.call(DICT_TH, key)) return DICT_TH[key];
    for (const [re, rep] of TH_PATTERNS) if (re.test(key)) return key.replace(re, rep);
    return null;
  }
  // which text nodes are worth looking at: Thai ones in English mode, Latin ones in Thai mode
  const NEEDS = lang === 'en' ? HAS_THAI : /[A-Za-z]/;

  function frag(s) { return s.replace(FRAG_RE, k => FRAGMENTS[k]); }
  // translation of a captured part; left as it is when it can't be translated
  function sub(x) { const t = trCore(x); return t == null ? x : t; }

  function trCore(s) {
    const key = norm(s);
    if (!key || !HAS_THAI.test(key)) return null;
    if (Object.prototype.hasOwnProperty.call(DICT, key)) return DICT[key];
    for (const [re, rep] of PATTERNS) {
      if (re.test(key)) return key.replace(re, rep);
    }
    // multi-line messages (alerts): translate line by line
    if (/\n/.test(s)) {
      const parts = s.split(/\n/);
      if (parts.length > 1) return parts.map(p => (p.trim() ? (trCore(p) ?? p) : p)).join('\n');
    }
    // only when EVERY Thai word is known — never mix languages inside
    // user text such as a name or a product description
    const f = frag(key);
    return f !== key && !HAS_THAI.test(f) ? f : null;
  }

  function tr(s) {
    if (typeof s !== 'string') return s;
    const out = lang === 'en' ? trCore(s) : trTh(s);
    if (out == null) return s;
    // keep the original's leading/trailing whitespace
    const lead = s.match(/^\s*/)[0], trail = s.match(/\s*$/)[0];
    return lead + out + trail;
  }

  // ---------------------------------------------------------------- DOM
  const ATTRS = ['placeholder', 'title', 'aria-label', 'alt'];

  function translateElementAttrs(el) {
    for (const a of ATTRS) {
      const v = el.getAttribute && el.getAttribute(a);
      if (v && NEEDS.test(v)) {
        const t = tr(v);
        if (t !== v) el.setAttribute(a, t);
      }
    }
  }

  function translateTree(root) {
    if (!root) return;
    if (root.nodeType === 3) {
      if (NEEDS.test(root.nodeValue)) {
        const t = tr(root.nodeValue);
        if (t !== root.nodeValue) root.nodeValue = t;
      }
      return;
    }
    if (root.nodeType !== 1 && root.nodeType !== 9 && root.nodeType !== 11) return;
    if (root.nodeType === 1) {
      const tag = root.tagName;
      if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'TEXTAREA') return;
      if (root.isContentEditable) return;
      translateElementAttrs(root);
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
      acceptNode(n) {
        if (n.nodeType === 1) {
          const tag = n.tagName;
          if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'TEXTAREA' || n.isContentEditable) return NodeFilter.FILTER_REJECT;
          translateElementAttrs(n);
          return NodeFilter.FILTER_SKIP;
        }
        return NEEDS.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(n => { const t = tr(n.nodeValue); if (t !== n.nodeValue) n.nodeValue = t; });
  }

  // ---------------------------------------------------------------- switch button
  function injectSwitch() {
    if (document.getElementById('langSwitch')) return;
    const css = document.createElement('style');
    css.textContent = `
      .lang-switch{display:inline-flex;border:1px solid var(--line,#E4DCEC);border-radius:999px;overflow:hidden;
        font:600 .72rem/1 Inter,system-ui,sans-serif;background:var(--panel,#fff);flex:none}
      .lang-switch button{border:0;background:transparent;color:var(--ink-soft,#7A6C8E);padding:6px 10px;cursor:pointer}
      .lang-switch button.on{background:var(--ink,#3B2A54);color:#fff}
      .lang-switch.floating{position:fixed;top:14px;right:14px;z-index:1500;box-shadow:0 4px 14px -6px rgba(36,26,51,.35)}`;
    document.head.appendChild(css);
    const box = document.createElement('div');
    box.id = 'langSwitch';
    box.className = 'lang-switch';
    box.setAttribute('role', 'group');
    box.setAttribute('aria-label', 'Language / ภาษา');
    box.innerHTML = '<button type="button" data-lang="th" lang="th">TH</button><button type="button" data-lang="en" lang="en">EN</button>';
    box.querySelectorAll('button').forEach(b => {
      b.classList.toggle('on', b.dataset.lang === lang);
      b.setAttribute('aria-pressed', String(b.dataset.lang === lang));
      b.addEventListener('click', () => I18N.setLang(b.dataset.lang));
    });
    // index.html: inside the header's user menu; login / reset pages: top-right corner
    const host = document.querySelector('.user-menu');
    if (host) host.insertBefore(box, host.firstChild);
    else { box.classList.add('floating'); document.body.appendChild(box); }
  }

  // ---------------------------------------------------------------- public API
  const I18N = {
    lang,
    locale: lang === 'en' ? 'en-GB' : 'th-TH',
    tr,
    setLang(l) {
      try { localStorage.setItem(STORE_KEY, l === 'en' ? 'en' : 'th'); } catch (e) { /* ignore */ }
      if (l !== lang) location.reload();
    },
    articles: {},
  };
  window.I18N = I18N;

  if (lang === 'en') {
    document.documentElement.lang = 'en';
    const _alert = window.alert.bind(window), _confirm = window.confirm.bind(window);
    window.alert = msg => _alert(tr(String(msg)));
    window.confirm = msg => _confirm(tr(String(msg)));
  }

  document.addEventListener('DOMContentLoaded', () => {
    injectSwitch();
    if (lang === 'en') document.title = tr(document.title);
    translateTree(document.body);
    new MutationObserver(muts => {
      for (const m of muts) {
        if (m.type === 'characterData') translateTree(m.target);
        else if (m.type === 'attributes') translateElementAttrs(m.target);
        else m.addedNodes.forEach(translateTree);
      }
    }).observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  });

  // ---------------------------------------------------------------- articles (English)
  I18N.articles = {
    'pore-size': {
      readTime: '4 min',
      title: 'How to make pores look smaller',
      teaser: 'Pores can\'t "shrink" for good, but the right ingredients and routine can make them look smaller.',
      body: `<p>Pore size is mostly set by genetics. What makes pores <em>look</em> bigger in the short term is excess oil, dead skin cells building up around the pore, and skin losing elasticity with age.</p>
    <p>What actually helps:</p>
    <ul>
      <li><strong>Control oil:</strong> niacinamide 5-10% reduces oil-gland activity with regular use.</li>
      <li><strong>Exfoliate regularly:</strong> salicylic acid (BHA) dissolves in oil, so it cleans deeper inside the pore than AHA.</li>
      <li><strong>Support elasticity:</strong> retinoids boost collagen around the pore, tightening its edge over time.</li>
      <li><strong>Sunscreen every day:</strong> UV breaks down collagen — the main reason pores sag and look larger with age.</li>
    </ul>
    <p>Avoid scrubbing hard too often: it makes the oil glands produce more oil to compensate, which makes things worse in the long run.</p>`,
    },
    'vitamin-c': {
      readTime: '5 min',
      title: 'Vitamin C and its derivatives: which one suits your skin',
      teaser: 'L-ascorbic acid is the strongest but irritates easily; derivatives such as MAP or THD ascorbate are gentler — here is how they differ.',
      body: `<p>Vitamin C is an antioxidant that slows pigment production and boosts collagen, but products use several forms with different strength and gentleness:</p>
    <ul>
      <li><strong>L-Ascorbic Acid:</strong> the pure form — strongest and fastest, but unstable, breaks down in light and air, and can irritate sensitive skin.</li>
      <li><strong>Sodium Ascorbyl Phosphate (SAP):</strong> more stable and gentle, good for oily and acne-prone skin, with a mild antibacterial effect.</li>
      <li><strong>Magnesium Ascorbyl Phosphate (MAP):</strong> water-soluble and gentle, good for sensitive skin, but less potent than the pure form.</li>
      <li><strong>Ascorbyl Glucoside / THD Ascorbate (Tetrahexyldecyl Ascorbate):</strong> very stable and absorbs well; common in newer oil-based serums.</li>
    </ul>
    <p>Tip: store vitamin C products in the dark in opaque bottles, and always pair them with sunscreen — vitamin C and SPF protect against free radicals better together.</p>`,
    },
    'aha-vs-bha': {
      readTime: '4 min',
      title: 'AHA vs BHA: what is the difference and which to choose',
      teaser: 'Both exfoliate, but they dissolve in different things, so they suit different skin concerns.',
      body: `<p><strong>AHA (Alpha Hydroxy Acid)</strong> such as glycolic and lactic acid dissolves in water and works mainly on the surface: it removes dull cells, fades shallow dark marks and boosts hydration. Good for dry, dull or uneven skin.</p>
    <p><strong>BHA (Beta Hydroxy Acid)</strong>, mainly salicylic acid, dissolves in oil, so it gets inside pores clogged with oil and dead cells. Good for oily skin, clogged pores and acne.</p>
    <p>In short: dry/dull skin → lean towards AHA; oily/acne-prone skin → lean towards BHA. Many people alternate both, but don't start them together because irritation is more likely. Begin with a low strength 2-3 times a week, and always wear sunscreen in the morning when using exfoliating acids.</p>`,
    },
    'azelaic-acid': {
      readTime: '3 min',
      title: '5 facts about azelaic acid',
      teaser: 'A natural acid found in wheat, rye and barley — gentle, yet it covers several skin concerns.',
      body: `<ol>
      <li><strong>Treats several concerns at once:</strong> acne, redness and post-acne marks, because it is antibacterial, anti-inflammatory and blocks the enzyme behind excess pigment.</li>
      <li><strong>Gentler than other exfoliating acids:</strong> at 10-20% it rarely causes strong peeling or stinging — good for sensitive skin and rosacea.</li>
      <li><strong>Usable during pregnancy:</strong> one of the few exfoliants dermatologists often consider safer than retinoids for pregnant women (always ask a doctor first).</li>
      <li><strong>Helps rosacea texture:</strong> clinical studies show it reduces redness and inflamed bumps in people with rosacea.</li>
      <li><strong>Needs time:</strong> most people need 8-12 weeks of regular use to see a clear change; it works gradually, not overnight.</li>
    </ol>`,
    },
    'niacinamide': {
      readTime: '3 min',
      title: 'What is niacinamide and why is it so popular',
      teaser: 'Vitamin B3 does many jobs in one bottle — from controlling oil to strengthening the skin barrier.',
      body: `<p>Niacinamide (vitamin B3) is one of the best-researched skincare ingredients because it works on several fronts at once:</p>
    <ul>
      <li>Controls sebum (oil), reducing shine and the look of large pores</li>
      <li>Strengthens the skin barrier by boosting the skin's own ceramides</li>
      <li>Fades acne marks and dark spots without irritation</li>
      <li>Calms mild redness and inflammation</li>
    </ul>
    <p>The usual strength is 5-10%, morning and evening. It works well with almost everything, including retinol and exfoliating acids, so it is a "basic" ingredient nearly any routine can include.</p>`,
    },
    'retinoids-101': {
      readTime: '5 min',
      title: 'Retinoids 101: retinol vs retinal vs tretinoin',
      teaser: 'All three belong to the same family, but how fast and how strongly they work differs a lot.',
      body: `<p>Retinoids are vitamin A derivatives that speed up cell turnover and boost collagen. The skin has to convert them before they work: the fewer conversion steps, the faster they act — and the more likely they are to irritate:</p>
    <ul>
      <li><strong>Retinol:</strong> sold over the counter; needs 2 conversion steps in the skin, so it is the gentlest but slowest. Good for beginners.</li>
      <li><strong>Retinal (retinaldehyde):</strong> only 1 step, faster and stronger than retinol, still no prescription needed.</li>
      <li><strong>Tretinoin:</strong> retinoic acid that works immediately without conversion — the strongest, and prescription-only in many countries because it irritates much more easily.</li>
    </ul>
    <p>General advice: start with low-strength retinol 2-3 nights a week, at night only, and always wear sunscreen in the morning — skin is more sensitive to the sun while using retinoids.</p>`,
    },
    'sunscreen-why': {
      readTime: '3 min',
      title: 'Why sunscreen is the most important step of any routine',
      teaser: 'However expensive your serum, skipping sunscreen is like fertilising a plant and then leaving it to burn in the sun every day.',
      body: `<p>UV is behind roughly 80% of premature skin ageing (photoaging) and also triggers pigment, making melasma, freckles and dark marks darker. Strong actives such as exfoliating acids or retinoids make skin even more sensitive to the sun.</p>
    <p>Choosing a sunscreen: SPF 30 or higher with broad-spectrum UVA/UVB protection. Apply enough (about two finger-lengths for the face) and reapply every 2-3 hours if you are outdoors for long. For oily or acne-prone skin, a light gel or non-comedogenic texture lowers the chance of breakouts from sunscreen.</p>`,
    },
    'centella': {
      readTime: '3 min',
      title: 'Centella asiatica for sensitive, delicate skin',
      teaser: 'A plant extract long used in traditional medicine, now backed by research for healing and calming the skin.',
      body: `<p>Centella asiatica is rich in triterpenoids such as madecassoside and asiaticoside, which calm inflammation, boost collagen and strengthen the skin barrier.</p>
    <p>It is especially good for easily irritated skin, skin that has just been through exfoliating acids or retinoids, or redness from inflamed acne, because it soothes without adding oil or clogging pores. It usually comes as creams, serums or cooling gel masks.</p>`,
    },
    'eyebag': {
      readTime: '4 min',
      title: 'Eye bags and dark circles: causes and what helps',
      teaser: 'The skin around the eyes is up to 5 times thinner than elsewhere on the face, so it needs different care.',
      body: `<p>Eye bags and dark circles usually have several causes together: lack of sleep makes blood pool under the thin skin around the eyes; genetics and age let the fat around the eyes shift into a bag; and collagen loss with age thins the skin so the vessels underneath show more.</p>
    <p>Ingredients with research behind them for this area:</p>
    <ul>
      <li><strong>Caffeine:</strong> boosts circulation and temporarily reduces puffiness; common in morning eye creams and gels.</li>
      <li><strong>Vitamin K + gentle retinol:</strong> helps dark circles caused by blood vessels and builds collagen around the eyes over time.</li>
      <li><strong>Peptides:</strong> strengthen the thin skin here without irritation.</li>
    </ul>
    <p>Things that help without any product: get enough sleep, raise the head of the bed slightly to reduce fluid build-up, and use a cold compress in the morning if you wake up puffy. Eye products should be extra light, because this skin is far more delicate and irritates more easily.</p>`,
    },
  };
})();
