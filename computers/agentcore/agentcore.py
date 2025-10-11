import termcolor
import json
import time
from ..playwright.playwright import PlaywrightComputer
from playwright.sync_api import sync_playwright


class AgentCoreComputer(PlaywrightComputer):
    """Connects to Amazon Bedrock AgentCore Browser via CDP.
    
    Supports optional session recording to S3 for replay and debugging.
    """

    def __init__(
        self,
        screen_size: tuple[int, int],
        initial_url: str = "https://www.google.com",
        recording_bucket: str = None,
        recording_prefix: str = "recordings",
        execution_role_arn: str = None,
        create_execution_role: bool = False,
    ):
        super().__init__(screen_size, initial_url)
        self._recording_bucket = recording_bucket
        self._recording_prefix = recording_prefix
        self._execution_role_arn = execution_role_arn
        self._create_execution_role = create_execution_role
        self._created_browser = False

    def _create_iam_role(self, region: str) -> str:
        """Create IAM role scoped to recording bucket."""
        import boto3
        import hashlib
        
        iam = boto3.client("iam")
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        
        # Create unique role name based on bucket/prefix
        bucket_prefix_hash = hashlib.sha256(
            f"{self._recording_bucket}/{self._recording_prefix}".encode()
        ).hexdigest()[:8]
        role_name = f"AgentCoreBrowserRecording-{bucket_prefix_hash}"
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account_id}:*"}
                }
            }]
        }
        
        role_created = False
        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Auto-created role for AgentCore browser recording"
            )
            role_created = True
            termcolor.cprint(f"Created IAM role: {role_name}", color="green")
        except iam.exceptions.EntityAlreadyExistsException:
            termcolor.cprint(f"Using existing IAM role: {role_name}", color="yellow")
        
        # Only update policy if we just created the role
        if role_created:
            permissions_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Action": ["s3:PutObject", "s3:ListMultipartUploadParts", "s3:AbortMultipartUpload"],
                    "Resource": f"arn:aws:s3:::{self._recording_bucket}/{self._recording_prefix}/*",
                    "Condition": {"StringEquals": {"aws:ResourceAccount": account_id}}
                }]
            }
            
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName="S3RecordingAccess",
                PolicyDocument=json.dumps(permissions_policy)
            )
            
            # Wait for IAM propagation
            termcolor.cprint("Waiting for IAM role to propagate...", color="yellow")
            time.sleep(10)
        
        return f"arn:aws:iam::{account_id}:role/{role_name}"

    def __enter__(self):
        from bedrock_agentcore.tools.browser_client import BrowserClient
        from boto3.session import Session
        import boto3
        from urllib.parse import urlencode

        print("Creating AgentCore browser session...")

        boto_session = Session()
        region = boto_session.region_name

        # Create browser with recording if bucket specified
        if self._recording_bucket:
            if not self._execution_role_arn and not self._create_execution_role:
                raise ValueError(
                    "execution_role_arn is required when using recording. "
                    "Pass --execution_role_arn or use --create_execution_role to auto-create."
                )
            
            # Auto-create role if requested
            if self._create_execution_role and not self._execution_role_arn:
                self._execution_role_arn = self._create_iam_role(region)
            
            control_client = boto3.client("bedrock-agentcore-control", region_name=region)
            
            import uuid
            browser_name = f"gemini_browser_{uuid.uuid4().hex[:12]}"
            
            browser_config = {
                "name": browser_name,
                "networkConfiguration": {"networkMode": "PUBLIC"},
                "executionRoleArn": self._execution_role_arn,
            }
            
            # Add recording configuration
            browser_config["recording"] = {
                "enabled": True,
                "s3Location": {
                    "bucket": self._recording_bucket,
                    "prefix": self._recording_prefix,
                }
            }
            
            response = control_client.create_browser(**browser_config)
            browser_id = response["browserId"]
            self._created_browser = True
            
            termcolor.cprint(
                f"Created browser {browser_id} with recording to s3://{self._recording_bucket}/{self._recording_prefix}",
                color="cyan",
            )

        self._client = BrowserClient(region)
        
        # TODO: When BrowserClient.start() supports viewport parameter, replace manual session start with:
        # self._client.start(
        #     identifier="aws.browser.v1",
        #     name="gemini-browser-session",
        #     session_timeout_seconds=3600,
        #     viewport={"width": self._screen_size[0], "height": self._screen_size[1]}
        # )
        
        # Start session with viewport using boto3 directly
        session_response = self._client.client.start_browser_session(
            browserIdentifier="aws.browser.v1",
            name="gemini-browser-session",
            sessionTimeoutSeconds=3600,
            viewPort={
                "width": self._screen_size[0],
                "height": self._screen_size[1]
            }
        )
        
        # Set client state
        self._client.identifier = session_response["browserIdentifier"]
        self._client.session_id = session_response["sessionId"]

        ws_url, headers = self._client.generate_ws_headers()

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.connect_over_cdp(
            ws_url,
            headers=headers
        )
        self._context = self._browser.contexts[0]
        self._page = self._context.pages[0]
        
        # Set viewport explicitly (CDP connection doesn't inherit from session config)
        self._page.set_viewport_size({
            "width": self._screen_size[0],
            "height": self._screen_size[1]
        })
        
        self._page.goto(self._initial_url)

        self._context.on("page", self._handle_new_page)

        termcolor.cprint(
            f"AgentCore browser session started in {region}",
            color="green",
            attrs=["bold"],
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._page:
            self._page.close()

        if self._context:
            self._context.close()

        if self._browser:
            self._browser.close()

        self._playwright.stop()

        if self._client:
            self._client.stop()
