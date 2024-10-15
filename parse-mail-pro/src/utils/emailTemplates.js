// src/utils/emailTemplates.js

const emailTemplates = {
    meeting: `Subject: Team Meeting - Project Update
  From: manager@company.com
  To: team@company.com
  Date: March 15, 2024
  
  Hi team,
  
  Let's meet to discuss the project progress. The meeting is scheduled for March 20, 2024 at 2:00 PM EST in Conference Room A.
  
  Agenda:
  1. Project timeline review
  2. Resource allocation
  3. Next steps
  
  Please confirm your attendance.
  
  Best regards,
  Manager`,
    invoice: `Subject: Invoice #INV-2024-001
  From: billing@supplier.com
  To: accounts@company.com
  Date: March 16, 2024
  
  Dear Customer,
  
  Please find attached invoice #INV-2024-001 for recent services.
  
  Amount Due: $1,500.00
  Due Date: March 30, 2024
  
  Payment Details:
  Bank: FirstBank
  Account: 1234567890
  Reference: INV-2024-001
  
  Thank you for your business!
  
  Regards,
  Billing Team`,
    shipping: `Subject: Your Order Has Shipped!
  From: orders@store.com
  To: customer@email.com
  Date: March 17, 2024
  
  Dear Customer,
  
  Your order #ORD123456 has shipped!
  
  Tracking Number: 1Z999AA1234567890
  Carrier: UPS
  Estimated Delivery: March 20, 2024
  
  Order Details:
  - Product A ($99.99)
  - Product B ($149.99)
  
  Track your package here: https://tracking.ups.com
  
  Thank you for shopping with us!
  
  Best regards,
  Store Team`,
    claim: `Subject: Insurance Claim Submission
  From: john.doe@example.com
  To: claims@insurancecompany.com
  Date: April 10, 2024
  
  Dear Claims Handler,
  
  I am writing to submit a claim regarding the recent loss at my property.
  
  **Requesting Party**
  Insurance Company: ABC Insurance
  Handler: Jane Smith
  Carrier Claim Number: CLM-2024-7890
  
  **Insured Information**
  Name: John Doe
  Contact #: (555) 123-4567
  Loss Address: 123 Elm Street, Springfield, IL
  Public Adjuster: Mark Johnson
  Is the insured an Owner or a Tenant of the loss location? Owner
  
  **Adjuster Information**
  Adjuster Name: Emily Davis
  Adjuster Phone Number: (555) 987-6543
  Adjuster Email: emily.davis@insurancecompany.com
  Job Title: Senior Adjuster
  Address: 456 Oak Avenue, Springfield, IL
  Policy #: POL-567890
  
  **Assignment Information**
  Date of Loss/Occurrence: April 5, 2024
  Cause of loss: Hail Storm
  Facts of Loss: On April 5th, a severe hail storm caused significant damage to the roof and siding of my home.
  Loss Description: Multiple shingles were torn off, and several sections of the siding are cracked or missing.
  Residence Occupied During Loss: Yes
  Was Someone home at time of damage: Yes
  Repair or Mitigation Progress: Initial assessments completed; awaiting approval for repairs.
  Type: Hail
  Inspection type: On-site inspection
  
  Check the box of applicable assignment type:
  - [x] Wind
  - [x] Structural
  - [x] Hail
  - [ ] Foundation
  - [ ] Other - provide details:
  
  **Additional details/Special Instructions:**
  Please expedite the inspection process as repairs are urgently needed to prevent further damage.
  
  **Attachment(s):**
  - Photos of damage
  - Initial repair estimates
  
  Thank you for your prompt attention to this matter.
  
  Sincerely,
  John Doe`,
  };
  
  export default emailTemplates;
  